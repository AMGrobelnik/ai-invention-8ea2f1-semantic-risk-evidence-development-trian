#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Standardize twitter-financial-news-sentiment dataset to exp_sel_data_out schema."""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path("/ai-inventor/aii_data/runs/run_PoDi6I8fYcAb/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
DATASETS_DIR = WORKSPACE / "temp" / "datasets"

TWITTER_LABEL_MAP = {0: "bearish", 1: "neutral", 2: "bullish"}


def load_twitter_sentiment() -> list[dict]:
    """Load all twitter financial news sentiment rows."""
    path = DATASETS_DIR / "full_zeroshot_twitter-financial-news-sentiment_default_train.json"
    logger.info(f"Loading twitter sentiment from {path}")
    raw = json.loads(path.read_text())
    logger.info(f"Loaded {len(raw)} twitter rows")
    examples = []
    for i, row in enumerate(raw):
        text = (row.get("text") or "").strip()
        label_int = row.get("label")
        if not text or label_int is None:
            continue
        label_str = TWITTER_LABEL_MAP.get(int(label_int), str(label_int))
        examples.append({
            "input": text,
            "output": label_str,
            "metadata_row_index": i,
            "metadata_task_type": "classification",
            "metadata_n_classes": 3,
            "metadata_label_int": label_int,
            "metadata_label_names": "bearish,neutral,bullish",
        })
    logger.info(f"Built {len(examples)} twitter examples")
    return examples


def main():
    twitter_examples = load_twitter_sentiment()

    output = {
        "metadata": {
            "description": "Twitter financial news sentiment dataset for SREDT experiment (supplementary)",
            "datasets_included": ["zeroshot/twitter-financial-news-sentiment"],
        },
        "datasets": [
            {
                "dataset": "zeroshot/twitter-financial-news-sentiment",
                "examples": twitter_examples,
            },
        ],
    }

    out_path = WORKSPACE / "full_data_out.json"
    logger.info(f"Writing {out_path}")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Done: {len(twitter_examples)} examples, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
