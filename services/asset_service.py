"""
Campaign output packaging.

Turns a CampaignResult (+ optional CampaignCopy) into the on-disk layout
described in project brief section 19:

    outputs/<campaign_slug>/
        campaign_brief.json
        strategy.json
        prompts/<asset_id>.txt
        images/<asset_id>.png          (the winning candidate for each asset)
        evaluations/<asset_id>.json
        campaign_copy.json             (only written if copy was generated)

Note: intermediate/failed candidates stay wherever
workflows.asset_generation wrote them (inside the working output_dir
passed to run_campaign) -- this function only copies the *selected* final
image into the clean, demo-ready campaign package.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pydantic import BaseModel

from schemas import CampaignCopy
from tools.file_tools import slugify
from workflows.campaign_workflow import CampaignResult

logger = logging.getLogger("Workflow")


def _write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2))


def _prompt_text(asset_prompt) -> str:
    return (
        f"POSITIVE:\n{asset_prompt.positive_prompt}\n\n"
        f"NEGATIVE:\n{asset_prompt.negative_prompt}\n\n"
        f"style: {asset_prompt.style}\n"
        f"lighting: {asset_prompt.lighting}\n"
        f"composition: {asset_prompt.composition}\n"
        f"aspect_ratio: {asset_prompt.aspect_ratio}\n"
    )


def package_campaign(
    result: CampaignResult,
    output_dir: str | Path,
    copy: CampaignCopy | None = None,
) -> Path:
    """Write the full campaign package to <output_dir>/<slug>/ and return
    that directory's path."""
    campaign_dir = Path(output_dir) / slugify(result.brief.campaign_name)
    images_dir = campaign_dir / "images"
    prompts_dir = campaign_dir / "prompts"
    evaluations_dir = campaign_dir / "evaluations"
    for d in (images_dir, prompts_dir, evaluations_dir):
        d.mkdir(parents=True, exist_ok=True)

    _write_json(campaign_dir / "campaign_brief.json", result.brief)
    _write_json(campaign_dir / "strategy.json", result.plan)

    for asset in result.assets:
        dest_image = images_dir / f"{asset.asset_id}.png"
        try:
            shutil.copyfile(asset.file_path, dest_image)
        except OSError as exc:
            logger.warning("Could not copy final image for asset '%s': %s", asset.asset_id, exc)

        (prompts_dir / f"{asset.asset_id}.txt").write_text(_prompt_text(asset.prompt))
        _write_json(evaluations_dir / f"{asset.asset_id}.json", asset.evaluation)

    if copy is not None:
        _write_json(campaign_dir / "campaign_copy.json", copy)

    logger.info("[Workflow] Campaign package written to %s", campaign_dir)
    return campaign_dir
