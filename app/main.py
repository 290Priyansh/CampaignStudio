"""
CLI entry point for the AI Creative Director.

Usage:
    python -m app.main "Create a marketing campaign for an eco-friendly \
running shoe targeted at college students aged 18-24." --assets 3

For a visual interface instead, run:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import setup_logging
from models.exceptions import CreativeDirectorError
from services.campaign_service import run_full_campaign

logger = logging.getLogger("App")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Creative Director end to end.")
    parser.add_argument("brief", help="The campaign brief text.")
    parser.add_argument("--assets", type=int, default=5, help="Number of assets to generate.")
    parser.add_argument("--output-dir", default=None, help="Override OUTPUT_DIR for this run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        result, copy, campaign_dir = run_full_campaign(
            args.brief, number_of_assets=args.assets, output_dir=args.output_dir
        )
    except CreativeDirectorError as exc:
        logger.error(str(exc))
        return 1

    logger.info(
        "Campaign '%s' complete: %d assets generated", result.brief.campaign_name, len(result.assets)
    )
    for asset in result.assets:
        status = "PASSED" if asset.evaluation.passed else "below threshold"
        logger.info(
            "  - %s: score=%.2f (%s), attempts=%d",
            asset.asset_id, asset.evaluation.overall_score, status, asset.attempts_used,
        )
    logger.info("Full package written to: %s", campaign_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
