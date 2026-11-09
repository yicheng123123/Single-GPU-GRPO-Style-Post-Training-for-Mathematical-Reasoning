
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# 复用之前已经完成的 CS336 A5 函数
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CS336 A5 Qwen SFT training"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/r1_sft/sft_128_math_train.jsonl",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/a5_sft_128_smoke",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-updates",
        type=int,
        default=2,
        help="0 means complete all epochs",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_records(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)

                if not isinstance(record.get("prompt"), str):
                    raise ValueError("Missing prompt")

                if not isinstance(record.get("response"), str):
                    raise ValueError("Missing response")

                records.append(record)

    if not records:
        raise ValueError("No training data found")

    return records


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        # 调用之前实现的 A5 Tokenization
        return run_tokenize_prompt_and_output(
            prompt_strs=[r["prompt"] for r in batch],
            output_strs=[r["response"] for r in batch],
            tokenizer=tokenizer,
        )

    return collate_fn


def main():
    args = parse_args()

    if args.grad_accum < 1:
        raise ValueError("grad-accum must be >= 1")

    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")

    if args.max_updates < 0:
        raise ValueError("max-updates must be >= 0")

    # -----------------------------
    # 1. 随机种子和 GPU 检查
    # -----------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm GPU is not available. "
            "Do not run full-parameter SFT on CPU."
        )

    device = torch.device("cuda:0")

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("ROCm version:", torch.version.hip)
    print("Device:", device, flush=True)

    # -----------------------------
    # 2. 输出目录及日志
    # -----------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "train_metrics.jsonl"
    config_path = output_dir / "run_config.json"

    config = vars(args).copy()
    config.update({
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "rocm_version": torch.version.hip,
        "dtype": "fp32_parameters_bf16_autocast",
        "microbatch_size": 1,
        "loss_normalize_constant": 1.0,
    })

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 每次运行单独使用一个 output 目录，
    # 这里清空该目录下旧的指标文件
    with metrics_path.open("w", encoding="utf-8"):
        pass

    # -----------------------------
    # 3. 读取 128 条 R1 SFT 数据
    # -----------------------------
    records = load_records(args.data)

    if len(records) % args.grad_accum != 0:
        raise ValueError(
            "For this experiment, dataset size must "
            "be divisible by grad-accum."
        )

    print("Training samples:", len(records))

    # -----------------------------
    # 4. Qwen Tokenizer + DataLoader
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    dataloader = DataLoader(
        records,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=make_collate_fn(tokenizer),
    )

    # -----------------------------
    # 5. 加载 Qwen2.5-Math-1.5B
    # -----------------------------
    print("Loading Qwen model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )

    model.to(device)

    # 训练阶段不使用 KV Cache
    model.config.use_cache = False

    # 节省激活显存
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    model.train()

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", trainable_params)

    # -----------------------------
    # 6. AdamW Optimizer
    # -----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    optimizer.zero_grad(set_to_none=True)

    # -----------------------------
    # 7. 初始化训练统计
    # -----------------------------
    global_step = 0
    micro_count = 0

    # A5 函数返回的 loss 已除以 grad_accum
    accumulated_loss = 0.0

    # Response Token NLL
    accumulated_nll_sum = 0.0
    accumulated_response_tokens = 0

    # Tokens/s 使用完整训练输入 token 数
    accumulated_input_tokens = 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    step_start = time.perf_counter()

    stop_training = False

    # -----------------------------
    # 8. 正式 SFT Training Loop
    # -----------------------------
    for epoch in range(args.epochs):

        print(
            f"\nStarting epoch {epoch + 1}/{args.epochs}",
            flush=True,
        )

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            if input_ids.shape[1] > args.max_seq_len:
                raise ValueError(
                    f"Sequence length {input_ids.shape[1]} "
                    f"exceeds {args.max_seq_len}"
                )

            response_token_count = int(
                response_mask.sum().item()
            )

            if response_token_count == 0:
                raise ValueError(
                    "Response mask contains no valid tokens"
                )

            # -------------------------
            # 8.1 Forward
            # -------------------------
            # Qwen 对 R1 Response 的逐 Token Log Probability
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                result = run_get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )

            log_probs = result["log_probs"].float()

            # -------------------------
            # 8.2 记录原始 Token NLL
            # -------------------------
            with torch.no_grad():
                response_nll_sum = -(
                    log_probs.detach() * response_mask
                ).sum()

                accumulated_nll_sum += (
                    response_nll_sum.item()
                )

            accumulated_response_tokens += (
                response_token_count
            )

            accumulated_input_tokens += (
                input_ids.numel()
            )

            # -------------------------
            # 8.3 SFT Loss + Backward
            # -------------------------
            # 注意：该函数内部已经 backward()
            # normalize_constant=1.0 与 A5 函数默认值一致
            loss, metadata = run_sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_accum,
                normalize_constant=1.0,
            )

            if not torch.isfinite(loss.detach()).item():
                raise RuntimeError(
                    "Non-finite SFT loss detected"
                )

            accumulated_loss += loss.detach().item()

            micro_count += 1

            # 释放当前 batch 的计算图引用
            del result, log_probs, loss

            # 未达到累积次数：继续处理下一道题
            if micro_count < args.grad_accum:
                continue

            # -------------------------
            # 8.4 Gradient Norm
            # -------------------------
            # 返回裁剪前的全局梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            if not torch.isfinite(grad_norm).item():
                raise RuntimeError(
                    "Non-finite gradient norm detected"
                )

            # -------------------------
            # 8.5 Optimizer Update
            # -------------------------
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # -------------------------
            # 8.6 时间与显存
            # -------------------------
            # GPU 异步运行，需要同步后计时
            torch.cuda.synchronize()

            elapsed = time.perf_counter() - step_start

            tokens_per_sec = (
                accumulated_input_tokens / max(elapsed, 1e-9)
            )

            peak_vram_gb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

            peak_reserved_gb = (
                torch.cuda.max_memory_reserved()
                / (1024 ** 3)
            )

            # 所有有效 Response Token 的平均 NLL
            token_nll = (
                accumulated_nll_sum
                / accumulated_response_tokens
            )

            # train_loss 是梯度累积后的 SFT Loss
            # 默认 normalize_constant=1.0，
            # 所以它是未按 Token 数平均的 Loss
            metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": accumulated_loss,
                "token_nll": token_nll,
                "grad_norm_pre_clip": grad_norm.item(),
                "peak_vram_gb": peak_vram_gb,
                "peak_reserved_gb": peak_reserved_gb,
                "tokens_per_sec": tokens_per_sec,
                "input_tokens": accumulated_input_tokens,
                "response_tokens": accumulated_response_tokens,
                "step_time_sec": elapsed,
                "lr": optimizer.param_groups[0]["lr"],
            }

            # -------------------------
            # 8.7 保存 JSONL 指标
            # -------------------------
            with metrics_path.open(
                "a", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(metrics, ensure_ascii=False)
                    + "\n"
                )

            print(
                f"Epoch {epoch + 1} | "
                f"Step {global_step} | "
                f"Loss {accumulated_loss:.4f} | "
                f"Token NLL {token_nll:.4f} | "
                f"Grad Norm {grad_norm.item():.4f} | "
                f"Peak VRAM {peak_vram_gb:.2f} GB | "
                f"Tokens/s {tokens_per_sec:.1f}",
                flush=True,
            )

            # -------------------------
            # 8.8 重置下一轮统计
            # -------------------------
            accumulated_loss = 0.0
            accumulated_nll_sum = 0.0
            accumulated_response_tokens = 0
            accumulated_input_tokens = 0
            micro_count = 0

            # -------------------------
            # 8.9 停止条件
            # -------------------------
            if (
                args.max_updates > 0
                and global_step >= args.max_updates
            ):
                stop_training = True
                break

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            step_start = time.perf_counter()

        if stop_training:
            break

    if micro_count != 0:
        raise RuntimeError(
            "Training ended with incomplete "
            "gradient accumulation."
        )

    # -----------------------------
    # 9. 保存模型 Checkpoint
    # -----------------------------
    print("\nSaving Qwen checkpoint...", flush=True)

    model.config.use_cache = True

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(output_dir)

    print("\nTraining finished")
    print("Optimizer updates:", global_step)
    print("Model checkpoint:", output_dir)
    print("Metrics file:", metrics_path)
    print("Config file:", config_path)


if __name__ == "__main__":
    main()
