"""Tests for dataset generation (minimal cells, determinism)."""

import pytest

from clift.common import TASKS
from clift.data import (
    _TASKS_REQUIRING_CLRS_SAMPLER,
    _clrs_sampler_available,
    generate_clift_dataset,
)


def test_generate_lookup_table_golden_cell() -> None:
    instances = generate_clift_dataset(
        n_instances_per_cell=1,
        seed=12345,
        tasks=["lookup_table"],
        formats=["demonstration"],
        applications=["forward"],
        difficulties=[1],
    )
    assert len(instances) == 1
    inst = instances[0]
    assert inst.task == "lookup_table"
    assert inst.format == "demonstration"
    assert inst.application == "forward"
    assert inst.difficulty == 1
    assert inst.instance_id == 0
    assert inst.target
    assert "mapping" in inst.latent_structure


def test_generate_affine_register_single_cell() -> None:
    instances = generate_clift_dataset(
        n_instances_per_cell=1,
        seed=999,
        tasks=["affine_dynamics_2d"],
        formats=["demonstration"],
        applications=["forward"],
        difficulties=[1],
    )
    assert len(instances) == 1
    assert instances[0].task == "affine_dynamics_2d"


def test_default_tasks_exclude_clrs_sampler_when_clrs_missing() -> None:
    if _clrs_sampler_available():
        pytest.skip("clrs is installed; default task list includes CLRS sampler tasks")

    instances = generate_clift_dataset(n_instances_per_cell=1, seed=1)
    tasks_in_run = {i.task for i in instances}
    assert not tasks_in_run & _TASKS_REQUIRING_CLRS_SAMPLER
    for t in TASKS:
        if t in _TASKS_REQUIRING_CLRS_SAMPLER:
            assert t not in tasks_in_run


@pytest.mark.clrs
def test_insertion_sort_requires_clrs_extra() -> None:
    if not _clrs_sampler_available():
        pytest.skip("clrs not installed")

    instances = generate_clift_dataset(
        n_instances_per_cell=1,
        seed=7,
        tasks=["insertion_sort"],
        formats=["demonstration"],
        applications=["forward"],
        difficulties=[1],
    )
    assert len(instances) == 1
    assert instances[0].task == "insertion_sort"
    assert "[" in instances[0].target or instances[0].target.strip("-").isdigit()


def test_explicit_clrs_task_without_package_errors() -> None:
    if _clrs_sampler_available():
        pytest.skip("clrs is installed")

    with pytest.raises(ValueError, match="optional CLRS"):
        generate_clift_dataset(
            n_instances_per_cell=1,
            seed=1,
            tasks=["insertion_sort"],
            formats=["demonstration"],
            applications=["forward"],
            difficulties=[1],
        )
