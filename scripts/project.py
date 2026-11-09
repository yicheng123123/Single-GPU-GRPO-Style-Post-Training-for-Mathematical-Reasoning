"""Project CLI; GPU operations require explicit flags and existing local inputs."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')
from grpo_posttraining.cli import main

if __name__ == '__main__':
    main()
