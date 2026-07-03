"""Report writers for America AI experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from america_ai.config import REPORTS_DIR, ensure_dirs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path, *, layer_rows: list[dict], best_config: dict | None, notes: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# America AI Summary",
        "",
        "America AI is an intentionally politically steered parody. Measured results are reported when available; unset fields mean the GPU/model run has not been completed.",
        "",
        "## Selected Layers",
    ]
    for row in layer_rows:
        if row.get("selected"):
            lines.append(
                f"- {row['concept']}: layer {row['layer']} score {row.get('layer_score', 0):.3f}, bootstrap {row.get('bootstrap_stability', 0):.3f}"
            )
    lines.extend(["", "## Best Configuration"])
    lines.append("```json")
    lines.append(
        json.dumps(best_config or {"status": "unset"}, indent=2, sort_keys=True)
    )
    lines.append("```")
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines) + "\n")


def initialize_reports(
    layer_rows: list[dict] | None = None, best_config: dict | None = None
) -> None:
    ensure_dirs()
    write_csv(REPORTS_DIR / "layer_metrics.csv", layer_rows or [])
    write_csv(REPORTS_DIR / "vector_cosines.csv", [])
    write_jsonl(REPORTS_DIR / "all_generations.jsonl", [])
    (REPORTS_DIR / "top_configs.md").write_text(
        "# Top America AI Configurations\n\nNo optimization run completed yet.\n"
    )
    write_summary(
        REPORTS_DIR / "summary.md",
        layer_rows=layer_rows or [],
        best_config=best_config,
        notes=[
            "Run `python america_harvest.py` on a machine with Gemma access to populate measured vectors.",
            "Run `python america_optimize.py --trials 120 --stage-one-prompts 8 --finalists 20 --seeds 0,1,2` after harvesting.",
        ],
    )
