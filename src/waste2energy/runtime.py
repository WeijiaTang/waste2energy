from __future__ import annotations

import os


_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_runtime_thread_defaults(default_threads: str = "1") -> None:
    """Apply conservative BLAS/OpenMP thread defaults unless the user opted out."""

    if str(os.environ.get("WTE_DISABLE_THREAD_GUARD", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return

    configured_threads = str(os.environ.get("WTE_DEFAULT_BLAS_THREADS", default_threads)).strip() or default_threads
    for env_name in _THREAD_ENV_VARS:
        os.environ.setdefault(env_name, configured_threads)
