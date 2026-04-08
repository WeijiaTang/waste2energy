from __future__ import annotations

import os

from waste2energy.runtime import apply_runtime_thread_defaults


def test_apply_runtime_thread_defaults_sets_conservative_defaults(monkeypatch):
    for env_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("WTE_DISABLE_THREAD_GUARD", raising=False)
    monkeypatch.delenv("WTE_DEFAULT_BLAS_THREADS", raising=False)

    apply_runtime_thread_defaults()

    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"


def test_apply_runtime_thread_defaults_respects_existing_values(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.delenv("WTE_DISABLE_THREAD_GUARD", raising=False)

    apply_runtime_thread_defaults()

    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert os.environ["OMP_NUM_THREADS"] == "6"
