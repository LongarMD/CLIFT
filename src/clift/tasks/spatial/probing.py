import random
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from .core import (
    SpatialGenerationError,
    answer_is_unique_under_evidence,
    apply_vector,
    is_identifiable_heuristic,
    shape,
    token_at,
    valid_start_ranges,
    vector_text,
)

_QUERY_MAX_ATTEMPTS = 220
_BASE_VECTOR_SAMPLES = 12
_OOD_MAX_MAG_BY_DIFF = {1: 3, 2: 4}


def set_last_query_diagnostics(
    held_out: Dict[str, Any],
    struct: Dict[str, Any],
    edge_evidence: List[Tuple[str, int, int, str]],
    rejection_counts: Dict[str, int],
    non_formal: bool,
    answer_unique_under_evidence: Any,
) -> None:
    """Store per-query diagnostics in the held_out metadata payload."""
    held_out["last_query_diagnostics"] = {
        "layout_identifiable": is_identifiable_heuristic(struct, edge_evidence)
        if edge_evidence
        else True,
        "answer_unique_under_evidence": answer_unique_under_evidence,
        "uniqueness_gate": "strict_nonformal" if non_formal else "formal_bypass",
        "edge_evidence_count": len(edge_evidence),
        "rejection_counts": rejection_counts,
    }


def query_failure_key(
    struct: Dict[str, Any],
    held_out: Dict[str, Any],
    edge_evidence: List[Tuple[str, int, int, str]],
    *,
    application: str,
    src: str,
    dst: str,
    dx: int,
    dy: int,
) -> str | None:
    """Return the first failed query gate key, or None when all gates pass."""
    if not token_visibility_ok(
        struct,
        held_out,
        application=application,
        start_token=src,
        end_token=dst,
        dx=dx,
        dy=dy,
    ):
        return "token_visibility"

    if edge_evidence and not is_identifiable_heuristic(struct, edge_evidence):
        return "layout_not_identifiable"

    answer_token = src if application != "inverse" else dst
    if edge_evidence and not answer_is_unique_under_evidence(
        struct,
        edge_evidence,
        application=application,
        token=answer_token,
        dx=dx,
        dy=dy,
    ):
        return "answer_not_unique_under_evidence"
    return None


def build_forward_question(src: str, dx: int, dy: int) -> str:
    """Build the canonical forward-style question prompt."""
    return (
        f"Question: Starting from '{src}', move {vector_text(dx, dy)}. "
        "Output only the resulting token.\n"
        "Final answer: "
    )


def sample_forward_inverse_vector(
    struct: Dict[str, Any], rng: random.Random
) -> Tuple[int, int]:
    """Forward and inverse base regime: local step-1 moves only."""
    _width, height = shape(struct)
    if height == 1:
        return rng.choice([-1, 1]), 0
    axis = rng.choice(["x", "y"])
    return (rng.choice([-1, 1]), 0) if axis == "x" else (0, rng.choice([-1, 1]))


def base_candidate_vectors(
    struct: Dict[str, Any], rng: random.Random
) -> List[Tuple[int, int]]:
    """Sample base vectors for forward and inverse probes."""
    return [
        sample_forward_inverse_vector(struct, rng) for _ in range(_BASE_VECTOR_SAMPLES)
    ]


def sample_valid_query(
    struct: Dict[str, Any],
    rng: random.Random,
    dx: int,
    dy: int,
) -> Tuple[str, str]:
    """Sample a valid start token and destination for a requested vector."""
    width, height = shape(struct)
    x_range, y_range = valid_start_ranges(width, height, dx, dy)
    x = rng.choice(list(x_range))
    y = rng.choice(list(y_range))
    start = token_at(struct, x, y)
    tx, ty = apply_vector(x, y, dx, dy)
    dst = token_at(struct, tx, ty)
    return start, dst


def query_path_tokens(
    struct: Dict[str, Any], start: str, dx: int, dy: int
) -> List[str]:
    """Enumerate tokens encountered when applying a vector from a start token."""
    coord = struct["token_map"][start]
    x = coord[0]
    y = coord[1] if len(coord) > 1 else 0
    toks = [start]
    sx = 1 if dx > 0 else -1
    for _ in range(abs(dx)):
        x += sx
        toks.append(token_at(struct, x, y))
    sy = 1 if dy > 0 else -1
    for _ in range(abs(dy)):
        y += sy
        toks.append(token_at(struct, x, y))
    return toks


def token_visibility_ok(
    struct: Dict[str, Any],
    held_out: Dict[str, Any],
    *,
    application: str,
    start_token: str,
    end_token: str,
    dx: int,
    dy: int,
) -> bool:
    """Check whether all evidence tokens needed for a query are visible in context."""
    if held_out.get("context_kind") == "formal_specification":
        return True
    visible = set(held_out.get("context_tokens", []))
    must_have = start_token if application != "inverse" else end_token
    if must_have not in visible:
        return False
    for tok in query_path_tokens(struct, start_token, dx, dy):
        if tok not in visible:
            return False
    return True


def sample_identifiable_query(
    struct: Dict[str, Any],
    held_out: Dict[str, Any],
    rng: random.Random,
    *,
    application: str,
    candidate_vectors: List[Tuple[int, int]],
    prefer_held_out: bool,
) -> Tuple[str, int, int, str]:
    """Sample a query that is answerable under current context evidence."""
    edge_evidence = held_out.get("context_edges", [])
    held_moves = held_out.get("held_out_moves", []) if prefer_held_out else []
    non_formal = held_out.get("context_kind") != "formal_specification"
    if non_formal and not edge_evidence:
        raise SpatialGenerationError(
            "Missing non-formal evidence edges for uniqueness-gated query sampling."
        )
    rej = Counter()
    for _ in range(_QUERY_MAX_ATTEMPTS):
        if held_moves:
            src, dx, dy, dst = rng.choice(held_moves)
        else:
            dx, dy = rng.choice(candidate_vectors)
            src, dst = sample_valid_query(struct, rng, dx, dy)
        failure_key = query_failure_key(
            struct,
            held_out,
            edge_evidence,
            application=application,
            src=src,
            dst=dst,
            dx=dx,
            dy=dy,
        )
        if failure_key is not None:
            rej[failure_key] += 1
            continue
        set_last_query_diagnostics(
            held_out=held_out,
            struct=struct,
            edge_evidence=edge_evidence,
            rejection_counts=dict(rej),
            non_formal=non_formal,
            answer_unique_under_evidence=True if edge_evidence else None,
        )
        return src, dx, dy, dst
    set_last_query_diagnostics(
        held_out=held_out,
        struct=struct,
        edge_evidence=edge_evidence,
        rejection_counts=dict(rej),
        non_formal=non_formal,
        answer_unique_under_evidence=False if edge_evidence else None,
    )
    raise SpatialGenerationError(f"Failed to sample identifiable {application} query.")


def probe_forward(
    struct: Dict[str, Any], held_out: Dict[str, Any], rng: random.Random
) -> Tuple[str, str]:
    """Build a forward probe prompt and expected target token."""
    src, dx, dy, dst = sample_identifiable_query(
        struct,
        held_out,
        rng,
        application="forward",
        candidate_vectors=base_candidate_vectors(struct, rng),
        prefer_held_out=True,
    )
    q = build_forward_question(src, dx, dy)
    return q, dst


def probe_inverse(
    struct: Dict[str, Any], held_out: Dict[str, Any], rng: random.Random
) -> Tuple[str, str]:
    """Build an inverse probe prompt and expected source token."""
    src, dx, dy, dst = sample_identifiable_query(
        struct,
        held_out,
        rng,
        application="inverse",
        candidate_vectors=base_candidate_vectors(struct, rng),
        prefer_held_out=True,
    )
    q = (
        f"Question: You moved {vector_text(dx, dy)} and ended up at '{dst}'. "
        "What was the starting token? Output only the starting token.\n"
        "Final answer: "
    )
    return q, src


def vector_candidates(
    struct: Dict[str, Any], predicate: Callable[[int, int], bool]
) -> List[Tuple[int, int]]:
    """Enumerate candidate vectors that satisfy a predicate."""
    width, height = shape(struct)
    max_step = max(width, height) - 1
    cands: List[Tuple[int, int]] = []
    for dx in range(-max_step, max_step + 1):
        for dy in range(-max_step, max_step + 1):
            if dx == 0 and dy == 0:
                continue
            if predicate(dx, dy):
                cands.append((dx, dy))
    return cands


def has_valid_start(struct: Dict[str, Any], dx: int, dy: int) -> bool:
    """Return whether at least one valid start exists for the vector."""
    width, height = shape(struct)
    x_range, y_range = valid_start_ranges(width, height, dx, dy)
    return x_range.start < x_range.stop and y_range.start < y_range.stop


def probe_ood(
    struct: Dict[str, Any],
    held_out: Dict[str, Any],
    rng: random.Random,
) -> Tuple[str, str, Dict[str, Any]]:
    """Build an out-of-distribution probe with novelty constraints."""
    context_vectors = {tuple(v) for v in held_out.get("context_vectors", [])}
    context_steps = set(held_out.get("context_step_magnitudes", []))
    context_composed = {tuple(v) for v in held_out.get("context_composed_vectors", [])}
    width, height = shape(struct)

    compositional = [
        (dx, dy)
        for dx, dy in vector_candidates(struct, lambda dx, dy: dx != 0 and dy != 0)
        if (dx, dy) not in context_vectors
        and (dx, dy) not in context_composed
        and has_valid_start(struct, dx, dy)
    ]
    max_mag = _OOD_MAX_MAG_BY_DIFF.get(struct["difficulty"], max(width, height) - 1)
    magnitude = [
        (dx, dy)
        for dx, dy in vector_candidates(struct, lambda dx, dy: (dx == 0) ^ (dy == 0))
        if max(abs(dx), abs(dy)) >= 2
        and max(abs(dx), abs(dy)) <= max_mag
        and max(abs(dx), abs(dy)) not in context_steps
        and has_valid_start(struct, dx, dy)
    ]

    subtype = choose_ood_subtype(struct, rng, compositional, magnitude)
    pool = compositional if subtype == "Compositional" else magnitude
    if not pool:
        raise SpatialGenerationError(
            "No valid OOD vectors available under novelty constraints."
        )
    src, dx, dy, dst = sample_identifiable_query(
        struct,
        held_out,
        rng,
        application="ood",
        candidate_vectors=pool,
        prefer_held_out=False,
    )
    question = build_forward_question(src, dx, dy)
    return question, dst, {"ood_subtype": subtype}


def choose_ood_subtype(
    struct: Dict[str, Any],
    rng: random.Random,
    compositional: List[Tuple[int, int]],
    magnitude: List[Tuple[int, int]],
) -> str:
    """Choose OOD subtype while respecting available novelty pools."""
    subtype = "Magnitude"
    if shape(struct)[1] > 1 and struct["difficulty"] >= 2:
        subtype = rng.choice(["Compositional", "Magnitude"])
    if subtype == "Compositional" and not compositional:
        subtype = "Magnitude"
    if subtype == "Magnitude" and not magnitude and compositional:
        subtype = "Compositional"
    return subtype


def probe_spatial_translation(
    struct: Dict[str, Any],
    held_out: Dict[str, Any],
    application: str,
    rng: random.Random,
) -> Tuple[str, str] | Tuple[str, str, Dict[str, Any]]:
    """Generate a probe question for spatial translation.

    Returns two fields for forward and inverse applications, and three fields
    for ood applications where subtype metadata is included.
    """
    probes = {
        "forward": probe_forward,
        "inverse": probe_inverse,
        "ood": probe_ood,
    }
    return probes[application](struct, held_out, rng)


__all__ = ["probe_spatial_translation"]
