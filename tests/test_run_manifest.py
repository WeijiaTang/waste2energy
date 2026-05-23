# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

from waste2energy.common.run_manifest import (
    build_reproducibility_manifest,
    file_digest,
    write_reproducibility_manifest,
)


def test_file_digest_records_hash_and_missing_files(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text("a,b\n1,2\n", encoding="utf-8")

    digest = file_digest("input.csv", root=tmp_path)
    missing = file_digest("missing.csv", root=tmp_path)

    assert digest.exists
    assert digest.path == "input.csv"
    assert digest.size_bytes == input_path.stat().st_size
    assert digest.sha256 is not None
    assert not missing.exists
    assert missing.sha256 is None


def test_build_reproducibility_manifest_records_inputs_outputs_and_runtime(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("a,b\n1,2\n", encoding="utf-8")
    output_path.write_text("score\n0.5\n", encoding="utf-8")

    manifest = build_reproducibility_manifest(
        command=["waste2energy-plan", "--scenario", "baseline"],
        inputs=["input.csv"],
        outputs=["output.csv"],
        parameters={"objective_weight_preset": "balanced"},
        root=tmp_path,
    )

    assert manifest["command"] == ["waste2energy-plan", "--scenario", "baseline"]
    assert manifest["parameters"]["objective_weight_preset"] == "balanced"
    assert manifest["inputs"][0]["sha256"]
    assert manifest["outputs"][0]["sha256"]
    assert manifest["runtime"]["python"]


def test_write_reproducibility_manifest_uses_stable_json(tmp_path):
    target = tmp_path / "manifests" / "run.json"
    payload = {"b": 2, "a": 1}

    written = write_reproducibility_manifest(target, payload)
    loaded = json.loads(written.read_text(encoding="utf-8"))

    assert written == target
    assert loaded == {"a": 1, "b": 2}

