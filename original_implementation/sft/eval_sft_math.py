
import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["base", "bf16", "fp32"],
        default=["base", "bf16", "fp32"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-root", default="evaluations")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.n > 0
    assert torch.cuda.is_available()

    base_path = Path("model_path.txt").read_text().strip()

    model_paths = {
        "base": base_path,
        "bf16": "checkpoints/a5_sft_128_full",
        "fp32": "checkpoints/a5_sft_128_fp32_full",
    }

    # 使用相同的 Qwen Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = Path(
        "cs336_alignment/prompts/r1_zero.prompt"
    ).read_text(encoding="utf-8")

    # 原始 MATH Test：不使用训练集
    dataset = load_dataset(
        "json",
        data_files="data/eval/math_test.jsonl",
        split="train",
    )

    if args.n > len(dataset):
        raise ValueError("Requested too many examples")

    # 固定并打乱题目顺序
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:args.n]

    output_dir = Path(args.output_root) / (
        f"math_test_n{args.n}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "eval_indices.json").open("w") as f:
        json.dump(indices, f)

    print("Evaluation examples:", len(indices))
    print("Indices:", indices[:10])

    summaries = {}

    # 逐个模型加载，避免同时占用显存
    for model_name in args.models:
        print(f"\nLoading {model_name}...", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_paths[model_name],
            dtype=torch.float32,
            attn_implementation="sdpa",
        ).to("cuda")

        model.eval()
        model.config.use_cache = True

        model_results = []
        result_path = output_dir / f"{model_name}.jsonl"

        start_time = time.perf_counter()

        with result_path.open("w", encoding="utf-8") as f:
            for count, idx in enumerate(indices, start=1):
                row = dataset[idx]

                question = row["problem"]
                ground_truth = str(row["answer"])
                prompt = template.replace(
                    "{question}", question
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to("cuda")

                prompt_length = inputs["input_ids"].shape[1]

                # 每道题的随机种子在三个模型间保持一致
                torch.manual_seed(args.seed + idx)
                torch.cuda.manual_seed_all(args.seed + idx)

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        generated = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            max_new_tokens=args.max_new_tokens,
                            stop_strings=["</answer>"],
                            tokenizer=tokenizer,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                response_ids = generated[0, prompt_length:]
                response = tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                )

                score = r1_zero_reward_fn(
                    response,
                    ground_truth,
                    fast=True,
                )

                record = {
                    "dataset_index": idx,
                    "model": model_name,
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response,
                    "generated_tokens": len(response_ids),
                    "format_reward": float(
                        score["format_reward"]
                    ),
                    "answer_reward": float(
                        score["answer_reward"]
                    ),
                    "reward": float(score["reward"]),
                }

                f.write(json.dumps(
                    record, ensure_ascii=False
                ) + "\n")
                f.flush()

                model_results.append(record)

                print(
                    f"{model_name} {count}/{args.n} "
                    f"answer={record['answer_reward']:.0f} "
                    f"format={record['format_reward']:.0f} "
                    f"tokens={record['generated_tokens']}",
                    flush=True,
                )

        elapsed = time.perf_counter() - start_time
        n = len(model_results)

        summary = {
            "model": model_name,
            "num_examples": n,
            "answer_accuracy": sum(
                r["answer_reward"] for r in model_results
            ) / n,
            "format_accuracy": sum(
                r["format_reward"] for r in model_results
            ) / n,
            "mean_generated_tokens": sum(
                r["generated_tokens"] for r in model_results
            ) / n,
            "elapsed_seconds": elapsed,
            "temperature": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        }

        summaries[model_name] = summary
        print("\nSummary:", summary, flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            summaries,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nEvaluation complete")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
