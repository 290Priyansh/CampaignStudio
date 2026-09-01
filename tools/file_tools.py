"""
Small filesystem-related utilities shared across the workflow and services
layers. Kept intentionally minimal per project brief section 31 ("do not
overengineer") -- this is not a general-purpose utils dump, just the one
piece of logic (slugifying a campaign name into a safe directory name)
that's needed in more than one place.
"""

from __future__ import annotations

import re


def slugify(name: str) -> str:
    """Turn a campaign name into a filesystem-safe, lowercase, underscore-separated slug."""
    slug = "".join(c.lower() if c.isalnum() else "_" for c in name)
    return re.sub(r"_+", "_", slug).strip("_")
