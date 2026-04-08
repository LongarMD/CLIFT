import importlib.util
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from .common import (
    APPLICATIONS,
    DIFFICULTIES,
    FAMILIES,
    FORMATS,
    TASKS,
    CLIFTInstance,
    applications_for_task,
    export_jsonl,
    load_jsonl,
)

# Import all task generators and formatters/probes
from .tasks import (
    format_affine_dynamics_2d,
    format_arithmetic,
    format_clrs_binary_search,
    format_clrs_insertion_sort,
    format_clrs_max_subarray,
    format_clrs_naive_string_matcher,
    format_conditional,
    format_lookup,
    format_register_machine_2d,
    format_spatial_translation,
    generate_affine_dynamics_2d,
    generate_arithmetic_rule,
    generate_clrs_binary_search,
    generate_clrs_insertion_sort,
    generate_clrs_max_subarray,
    generate_clrs_naive_string_matcher,
    generate_conditional_rule,
    generate_lookup_table,
    generate_register_machine_2d,
    generate_spatial_translation,
    probe_affine_dynamics_2d,
    probe_arithmetic,
    probe_clrs_binary_search,
    probe_clrs_insertion_sort,
    probe_clrs_max_subarray,
    probe_clrs_naive_string_matcher,
    probe_conditional,
    probe_lookup,
    probe_register_machine_2d,
    probe_spatial_translation,
)
from .tasks.spatial import SpatialGenerationError

# CLRS algorithmic tasks now use a dedicated axis space.
CLRS_FORMATS = [
    "demonstration",
    "declarative_natural_language",
    "trace",
    "formal_specification",
]
CLRS_APPLICATIONS = [
    "forward",
    "inverse",
    "ood",
]

# ---------------------------------------------------------------------------
# Task generator dispatch
# ---------------------------------------------------------------------------


# We normalize formatter signatures so they all accept:
#   (struct, format, application, rng) -> (context, held_out)
# For non-CLRS tasks we ignore `application`; CLRS formatters use it.
def _fmt_ignore_app(formatter):
    return lambda struct, fmt, app, rng: formatter(struct, fmt, rng)


_TASK_GENERATORS = {
    "lookup_table": generate_lookup_table,
    "arithmetic_rule": generate_arithmetic_rule,
    "conditional_rule": generate_conditional_rule,
    "insertion_sort": generate_clrs_insertion_sort,
    "max_subarray": generate_clrs_max_subarray,
    "binary_search": generate_clrs_binary_search,
    "naive_string_matcher": generate_clrs_naive_string_matcher,
    "spatial_translation": generate_spatial_translation,
    "affine_dynamics_2d": generate_affine_dynamics_2d,
    "register_machine_2d": generate_register_machine_2d,
}

_TASK_FORMATTERS = {
    "lookup_table": _fmt_ignore_app(format_lookup),
    "arithmetic_rule": _fmt_ignore_app(format_arithmetic),
    "conditional_rule": _fmt_ignore_app(format_conditional),
    "insertion_sort": format_clrs_insertion_sort,
    "max_subarray": format_clrs_max_subarray,
    "binary_search": format_clrs_binary_search,
    "naive_string_matcher": format_clrs_naive_string_matcher,
    "spatial_translation": format_spatial_translation,
    "affine_dynamics_2d": _fmt_ignore_app(format_affine_dynamics_2d),
    "register_machine_2d": _fmt_ignore_app(format_register_machine_2d),
}

_TASK_PROBES = {
    "lookup_table": probe_lookup,
    "arithmetic_rule": probe_arithmetic,
    "conditional_rule": probe_conditional,
    "insertion_sort": probe_clrs_insertion_sort,
    "max_subarray": probe_clrs_max_subarray,
    "binary_search": probe_clrs_binary_search,
    "naive_string_matcher": probe_clrs_naive_string_matcher,
    "spatial_translation": probe_spatial_translation,
    "affine_dynamics_2d": probe_affine_dynamics_2d,
    "register_machine_2d": probe_register_machine_2d,
}

# Tasks that use CLRS-style format names and application axes (forward / inverse / ood).
# Includes ``spatial_translation``, which is not CLRS-backed but shares the same axes.
_CLRS_FORMAT_AXIS_TASKS: Set[str] = {
    "insertion_sort",
    "max_subarray",
    "binary_search",
    "naive_string_matcher",
    "spatial_translation",
}

# Tasks whose generators call into the optional ``clrs`` Python package (dm-clrs).
_TASKS_REQUIRING_CLRS_SAMPLER: Set[str] = {"insertion_sort", "binary_search"}


def _clrs_sampler_available() -> bool:
    return importlib.util.find_spec("clrs") is not None


def _task_default_formats(task: str) -> List[str]:
    return CLRS_FORMATS if task in _CLRS_FORMAT_AXIS_TASKS else list(FORMATS)


def _task_default_applications(task: str) -> List[str]:
    if task in _CLRS_FORMAT_AXIS_TASKS:
        return CLRS_APPLICATIONS
    return list(applications_for_task(task))


# ---------------------------------------------------------------------------
# Unsupported (task, format, application) combinations
# ---------------------------------------------------------------------------

# Maps task -> set of (format, application) pairs that are NOT supported.
# If a task is not in this dict, all combinations are supported.
_UNSUPPORTED_COMBOS: Dict[str, Set[Tuple[str, str]]] = {}


def _is_supported(task: str, fmt: str, application: str) -> bool:
    """Check if a (task, format, application) combination is supported."""
    unsupported = _UNSUPPORTED_COMBOS.get(task)
    if unsupported is None:
        return True
    return (fmt, application) not in unsupported


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def _generate_instance(
    task: str,
    fmt: str,
    application: str,
    difficulty: int,
    seed: int,
    instruct: bool = False,
) -> CLIFTInstance:
    """Deterministically generate a single CLIFT instance from a seed."""
    rng = random.Random(seed)
    last_exc: Exception | None = None
    for _attempt in range(12):
        try:
            struct = _TASK_GENERATORS[task](
                seed=rng.randint(0, 2**31),
                difficulty=difficulty,
            )

            context, held_out = _TASK_FORMATTERS[task](
                struct,
                fmt,
                application,
                random.Random(rng.randint(0, 2**31)),
            )

            probe_out = _TASK_PROBES[task](
                struct, held_out, application, random.Random(rng.randint(0, 2**31))
            )
            metadata: Dict[str, Any] = {}
            if isinstance(probe_out, tuple) and len(probe_out) == 3:
                question, target, metadata = probe_out
            else:
                question, target = probe_out

            if isinstance(held_out, dict):
                context_diag = held_out.get("context_diagnostics")
                query_diag = held_out.get("last_query_diagnostics")
                if context_diag is not None:
                    metadata["context_diagnostics"] = context_diag
                if query_diag is not None:
                    metadata["query_diagnostics"] = query_diag

            prompt = context + question

            messages = None
            if instruct:
                messages = [
                    {
                        "role": "user",
                        "content": context.strip() + "\n\n" + question.strip(),
                    },
                ]

            serialisable = {k: v for k, v in struct.items() if not k.startswith("_")}

            return CLIFTInstance(
                task=task,
                format=fmt,
                application=application,
                difficulty=difficulty,
                prompt=prompt,
                target=target,
                latent_structure=serialisable,
                instruct=instruct,
                messages=messages,
                metadata=metadata,
            )
        except SpatialGenerationError as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"Failed to generate valid instance for task={task}, format={fmt}, "
        f"application={application}, difficulty={difficulty} after retries: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_clift_dataset(
    n_instances_per_cell: int = 10,
    seed: int = 42,
    tasks: Optional[List[str]] = None,
    formats: Optional[List[str]] = None,
    applications: Optional[List[str]] = None,
    difficulties: Optional[List[int]] = None,
    instruct: bool = False,
) -> List[CLIFTInstance]:
    """Generate the complete CLIFT evaluation matrix.

    Creates n_instances_per_cell instances for every
    (task, format, application, difficulty) combination.

    Args:
        n_instances_per_cell: Number of instances per (T, F, A, D) cell.
        seed: Master random seed for reproducibility.
        tasks: Subset of tasks (default: all tasks in ``TASKS``, except that
            ``insertion_sort`` and ``binary_search`` are omitted if the optional
            ``clrs`` package is not installed unless you pass them explicitly).
        formats: Subset of formats (default: all four).
        applications: Subset of applications (default: union of all task apps).
        difficulties: Subset of difficulty levels (default: [1, 2, 3]).
        instruct: If True, populate messages on each instance for use
            with tokenizer.apply_chat_template().

    Returns:
        List of CLIFTInstance objects.
    """
    rng = random.Random(seed)
    if tasks is None:
        tasks = (
            list(TASKS)
            if _clrs_sampler_available()
            else [t for t in TASKS if t not in _TASKS_REQUIRING_CLRS_SAMPLER]
        )
    else:
        tasks = list(tasks)

    if not _clrs_sampler_available():
        blocked = [t for t in tasks if t in _TASKS_REQUIRING_CLRS_SAMPLER]
        if blocked:
            raise ValueError(
                f"These tasks require the optional CLRS dependency: {blocked}. "
                "Install with pip install 'clift[clrs]' or uv sync --extra clrs."
            )

    difficulties = difficulties or DIFFICULTIES

    missing = [t for t in tasks if t not in _TASK_GENERATORS]
    if missing:
        available = sorted(_TASK_GENERATORS.keys())
        raise ValueError(
            f"Unsupported tasks requested: {missing}. "
            f"Available tasks in this build: {available}"
        )

    instances: List[CLIFTInstance] = []
    idx = 0
    for task in tasks:
        task_formats = (
            list(formats) if formats is not None else _task_default_formats(task)
        )
        task_apps = (
            list(applications)
            if applications is not None
            else _task_default_applications(task)
        )

        supported_formats = set(_task_default_formats(task))
        supported_apps = set(_task_default_applications(task))
        bad_formats = [f for f in task_formats if f not in supported_formats]
        bad_apps = [a for a in task_apps if a not in supported_apps]
        if bad_formats or bad_apps:
            raise ValueError(
                f"Task '{task}' received unsupported axes. "
                f"Unsupported formats={bad_formats}, unsupported applications={bad_apps}. "
                f"Supported formats={sorted(supported_formats)}, "
                f"supported applications={sorted(supported_apps)}"
            )

        for fmt in task_formats:
            for app in task_apps:
                if not _is_supported(task, fmt, app):
                    continue
                for diff in difficulties:
                    for _ in range(n_instances_per_cell):
                        inst = _generate_instance(
                            task,
                            fmt,
                            app,
                            diff,
                            seed=rng.randint(0, 2**31),
                            instruct=instruct,
                        )
                        inst.instance_id = idx
                        instances.append(inst)
                        idx += 1
    return instances


__all__ = [
    "APPLICATIONS",
    "DIFFICULTIES",
    "FAMILIES",
    "FORMATS",
    "TASKS",
    "CLIFTInstance",
    "export_jsonl",
    "generate_clift_dataset",
    "load_jsonl",
]
