# Copyright 2026 Entalpic
"""Parallel reads and writes must keep working in processes forked after the
parent already used them (e.g. DataLoader workers with the default `fork`)."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import atompack
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork") or sys.platform == "darwin",
    reason="requires the fork start method",
)

N_MOLECULES = 64


def _write_db(path: Path) -> None:
    rng = np.random.default_rng(0)
    db = atompack.Database(str(path), compression="zstd")
    db.add_arrays_batch(
        rng.random((N_MOLECULES, 4, 3), dtype=np.float32),
        rng.integers(1, 80, (N_MOLECULES, 4), dtype=np.uint8),
        energy=np.arange(N_MOLECULES, dtype=np.float64),
    )
    db.flush()


def _child_reads(path: str) -> list[float]:
    db = atompack.Database.open(path)
    indices = list(range(N_MOLECULES))
    flat = db.get_molecules_flat(indices)
    molecules = db.get_molecules(indices)
    assert [mol.energy for mol in molecules] == flat["energy"].tolist()
    return flat["energy"].tolist()


def _child_writes(path: str) -> int:
    _write_db(Path(path))
    return len(atompack.Database.open(path))


def _run_in_forked_child(fn, arg: str):
    with mp.get_context("fork").Pool(1) as pool:
        # A deadlocked child surfaces as a TimeoutError instead of hanging the
        # suite; leaving the context manager terminates the worker.
        return pool.apply_async(fn, (arg,)).get(timeout=60)


@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_batch_reads_in_child_forked_after_parent_read(tmp_path: Path) -> None:
    path = tmp_path / "data.atp"
    _write_db(path)
    expected = _child_reads(str(path))  # starts the parent's thread pool

    assert _run_in_forked_child(_child_reads, str(path)) == expected


@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_batch_writes_in_child_forked_after_parent_write(tmp_path: Path) -> None:
    _write_db(tmp_path / "parent.atp")  # starts the parent's thread pool

    assert _run_in_forked_child(_child_writes, str(tmp_path / "child.atp")) == N_MOLECULES
