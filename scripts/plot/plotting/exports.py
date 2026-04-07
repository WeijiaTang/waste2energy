from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.plot.common import EXPORT_FORMATS, ensure_results_dir


def save_plot_figure_set(fig, stem: str, *, output_dir: Path) -> dict[str, str]:
    target_dir = ensure_results_dir(output_dir)
    outputs: dict[str, str] = {}
    for extension in EXPORT_FORMATS:
        format_dir = target_dir / extension
        path = format_dir / f"{stem}.{extension}"
        save_kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
            "facecolor": "white",
        }
        if extension in {"png", "tiff"}:
            save_kwargs["dpi"] = 600
        if extension == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format=extension, **save_kwargs)
        outputs[extension] = str(path)
    return outputs


def build_plot_manifest(
    *,
    outputs: dict[str, dict[str, str]],
    data_outputs: dict[str, str],
    output_dir: Path,
) -> dict[str, object]:
    return {
        "output_dir": str(output_dir),
        "data_outputs": data_outputs,
        "figure_outputs": outputs,
        "latex_pdf_targets": {
            figure_id: figure_outputs.get("pdf")
            for figure_id, figure_outputs in outputs.items()
        },
    }
