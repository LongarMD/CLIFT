import copy
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np


def _clrs_build_sampler(*args: Any, **kwargs: Any):
    """Lazily load DeepMind CLRS (optional dependency via ``clift[clrs]``)."""
    try:
        from clrs import build_sampler as build_clrs_sampler
    except ImportError as e:
        raise ImportError(
            "Tasks 'insertion_sort' and 'binary_search' require the optional CLRS "
            "dependency. Install with: pip install 'clift[clrs]' "
            "(or uv sync --extra clrs)."
        ) from e
    return build_clrs_sampler(*args, **kwargs)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _predecessors_to_order(preds: np.ndarray) -> List[int]:
    preds = preds.astype(int)
    y = np.ones(len(preds))
    y[preds] = 0
    [last] = np.where(y)[0]
    order = np.zeros(len(preds), dtype=int)
    order[-1] = last
    for i in range(len(order) - 2, -1, -1):
        order[i] = preds[order[i + 1]]
    return order.tolist()


def _apply_order(values: List[float], order: List[int]) -> List[float]:
    return [values[i] for i in order]


# ============================================================================
# TASK 1: CLRS INSERTION SORT
# ============================================================================

CLRS_ISORT_DIFFICULTY = {
    1: {"length": 5, "n_fs_examples": 3},
    2: {"length": 8, "n_fs_examples": 4},
    3: {"length": 12, "n_fs_examples": 5},
}


def generate_clrs_insertion_sort(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    cfg = CLRS_ISORT_DIFFICULTY[difficulty]
    length = cfg["length"]

    sampler, spec = _clrs_build_sampler(
        name="insertion_sort",
        seed=seed,
        num_samples=1,
        length=length,
        track_max_steps=True,
    )

    sample = sampler.next(batch_size=1)

    raw_inputs = next(f.data[0] for f in sample.features.inputs if f.name == "key")
    inputs = [round(x * 100) for x in raw_inputs.tolist()]

    raw_preds = next(f.data[0] for f in sample.outputs if f.name == "pred")
    final_order = _predecessors_to_order(raw_preds)
    final_sorted = _apply_order(inputs, final_order)

    raw_hints = next(f.data for f in sample.features.hints if f.name == "pred_h")

    trace_states = []
    last_state = None
    for step_pointers in raw_hints:
        step_order = _predecessors_to_order(step_pointers[0])
        step_state = _apply_order(inputs, step_order)
        if step_state != last_state:
            trace_states.append(step_state)
            last_state = step_state

    return {
        "task_type": "insertion_sort",
        "difficulty": difficulty,
        "inputs": inputs,
        "final_sorted": final_sorted,
        "final_order": final_order,
        "trace_states": trace_states,
        "seed": seed,
    }


# --- FORMATTERS ---


def _clrs_isort_format_demonstration(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    cfg = CLRS_ISORT_DIFFICULTY[struct["difficulty"]]
    sampler, _ = _clrs_build_sampler(
        name="insertion_sort",
        seed=struct["seed"] + 1,
        num_samples=cfg["n_fs_examples"],
        length=cfg["length"],
        track_max_steps=False,
    )

    lines = []
    for _ in range(cfg["n_fs_examples"]):
        sample = sampler.next(batch_size=1)
        fs_in = [
            round(x * 100)
            for x in next(
                f.data[0] for f in sample.features.inputs if f.name == "key"
            ).tolist()
        ]
        fs_preds = next(f.data[0] for f in sample.outputs if f.name == "pred")
        fs_order = _predecessors_to_order(fs_preds)
        fs_out = _apply_order(fs_in, fs_order)
        lines.append(f"Input: {fs_in}\nOutput: {fs_out}")

    lines.append(f"Target array: {struct['inputs']}")
    return "\n\n".join(lines) + "\n\n", []


def _clrs_isort_format_declarative_natural_language(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    return (
        f"Rule: Rearrange the elements in the array into strictly ascending order.\n\nTarget array: {struct['inputs']}\n\n",
        [],
    )


def _clrs_isort_format_trace(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    trace = struct["trace_states"]
    lines = [
        f"Target array: {struct['inputs']}",
        "Execution trace (state of array after each sequential placement):",
    ]
    for i, state in enumerate(trace):
        lines.append(f"Step {i}: {state}")
    return "\n".join(lines) + "\n\n", []


def _clrs_isort_format_formal_specification(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    math_spec = "Formal Specification:\n∀ i ∈ [1, N-1]:\n  1. Extract A[i]\n  2. Shift A[j] to A[j+1] for j from i-1 down to 0 until A[j] ≤ A[i] or j < 0\n  3. Insert A[i] at the freed position\n\n"
    data = {"A": struct["inputs"]}  # Matched variable A
    return math_spec + "Data:\n```json\n" + json.dumps(data) + "\n```\n\n", []


def format_clrs_insertion_sort(
    struct: Dict[str, Any], fmt: str, app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    struct["current_format"] = fmt
    formatters = {
        "demonstration": _clrs_isort_format_demonstration,
        "declarative_natural_language": _clrs_isort_format_declarative_natural_language,
        "trace": _clrs_isort_format_trace,
        "formal_specification": _clrs_isort_format_formal_specification,
    }
    return formatters[fmt](struct, app, rng)


# --- PROBES ---


# UPGRADED: Removed explicit rule definitions ("sort into ascending order"). Now pure structure application.
def _clrs_isort_probe_forward(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    q = "Question: Apply the latent structure defined in the context above to the Target Array. Answer with the complete final list in form [item1, item2, ...]. One line only.\nAnswer: "
    return q, str(struct["final_sorted"])


def _clrs_isort_probe_inverse(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    target_idx = rng.randint(0, len(struct["final_sorted"]) - 1)
    target_val = struct["final_sorted"][target_idx]

    # UPGRADED: Use the true mathematical permutation instead of list.index()
    orig_idx = struct["final_order"][target_idx]

    q = f"Question: Apply the latent structure defined in the context above. In the final output, the element '{target_val}' ends up at 0-indexed position {target_idx}. At what exact 0-indexed position did this specific element originate in the original Target Array? Give only the exact integer.\nAnswer: "
    return q, str(orig_idx)


def _clrs_isort_probe_ood(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    ood_length = len(struct["inputs"]) * 2
    ood_array = sorted(
        [rng.randint(-500, 500) for _ in range(ood_length)], reverse=True
    )

    q = f"Question: Apply the latent structure defined in the context above to this entirely new, adversarial Target Array: {ood_array}. Answer with the complete final list in form [item1, item2, ...]. One line only.\nAnswer: "
    return q, str(sorted(ood_array))


def probe_clrs_insertion_sort(
    struct: Dict[str, Any], held_out: List[Any], application: str, rng: random.Random
) -> Tuple[str, str]:
    probes = {
        "forward": _clrs_isort_probe_forward,
        "inverse": _clrs_isort_probe_inverse,
        "ood": _clrs_isort_probe_ood,
    }
    return probes[application](struct, held_out, rng)


# ============================================================================
# TASK 2: CLRS MAXIMUM SUBARRAY
# ============================================================================

CLRS_MAXSUB_DIFFICULTY = {
    1: {"length": 6, "n_fs_examples": 3},
    2: {"length": 10, "n_fs_examples": 4},
    3: {"length": 16, "n_fs_examples": 5},
}


def generate_clrs_max_subarray(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    rng = random.Random(seed)
    cfg = CLRS_MAXSUB_DIFFICULTY[difficulty]

    inputs = [rng.randint(-100, 100) for _ in range(cfg["length"])]

    trace_states = []
    best_sum = float("-inf")
    current_sum = 0
    best_l, best_h = 0, 0
    current_l = 0

    for i, val in enumerate(inputs):
        if current_sum <= 0:
            current_l = i
            current_sum = val
        else:
            current_sum += val

        if current_sum > best_sum:
            best_sum = current_sum
            best_l = current_l
            best_h = i

        trace_states.append({"current": [current_l, i], "best": [best_l, best_h]})

    return {
        "task_type": "max_subarray",
        "difficulty": difficulty,
        "inputs": inputs,
        "final_span": [best_l, best_h],
        "max_sum": best_sum,
        "trace_states": trace_states,
        "seed": seed,
    }


# --- FORMATTERS ---


def _clrs_maxsub_format_demonstration(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    cfg = CLRS_MAXSUB_DIFFICULTY[struct["difficulty"]]
    lines = []
    for i in range(cfg["n_fs_examples"]):
        fs_rng = random.Random(struct["seed"] + i + 100)
        fs_in = [fs_rng.randint(-100, 100) for _ in range(cfg["length"])]

        best_sum, c_sum, best_l, best_h, c_l = float("-inf"), 0, 0, 0, 0
        for j, val in enumerate(fs_in):
            if c_sum <= 0:
                c_l = j
                c_sum = val
            else:
                c_sum += val
            if c_sum > best_sum:
                best_sum = c_sum
                best_l = c_l
                best_h = j

        lines.append(f"Array: {fs_in}\nOutput Bounds: [{best_l}, {best_h}]")

    lines.append(f"Target array: {struct['inputs']}")
    return "\n\n".join(lines) + "\n\n", []


def _clrs_maxsub_format_declarative_natural_language(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    # UPGRADED: Explicitly define the left-most tie-breaker for identical max sums
    return (
        f"Rule: Find the contiguous sequence of numbers within the array that yields the highest possible sum when added together. If there are multiple sequences tied for the maximum sum, identify the left-most (earliest) sequence. Output the 0-indexed start and end bounds.\n\nTarget array: {struct['inputs']}\n\n",
        [],
    )


def _clrs_maxsub_format_trace(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    trace = struct["trace_states"]
    lines = [
        f"Target array: {struct['inputs']}",
        "Execution trace (tracked bounds at each step):",
    ]
    for i, state in enumerate(trace):
        lines.append(
            f"Step {i}: Current running bounds {state['current']}, Best bounds so far {state['best']}"
        )
    return "\n".join(lines) + "\n\n", []


def _clrs_maxsub_format_formal_specification(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    math_spec = "Formal Specification:\nInitialize S = 0, Best = -∞, L_curr = 0, L_best = 0, R_best = 0\n∀ i ∈ [0, N-1]:\n  if S ≤ 0: S = A[i]; L_curr = i\n  else: S = S + A[i]\n  if S > Best: Best = S; L_best = L_curr; R_best = i\n\n"
    data = {"A": struct["inputs"]}  # Matched variable A
    return math_spec + "Data:\n```json\n" + json.dumps(data) + "\n```\n\n", []


def format_clrs_max_subarray(
    struct: Dict[str, Any], fmt: str, app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    struct["current_format"] = fmt
    formatters = {
        "demonstration": _clrs_maxsub_format_demonstration,
        "declarative_natural_language": _clrs_maxsub_format_declarative_natural_language,
        "trace": _clrs_maxsub_format_trace,
        "formal_specification": _clrs_maxsub_format_formal_specification,
    }
    return formatters[fmt](struct, app, rng)


# --- PROBES ---


# UPGRADED: Rule-agnostic.
def _clrs_maxsub_probe_forward(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    q = "Question: Apply the latent structure defined in the context above to the Target Array. Identify the 0-indexed [start, end] bounds of the optimal sequence. Answer with the list in form [start, end]. One line only.\nAnswer: "
    return q, str(struct["final_span"])


def _clrs_maxsub_probe_inverse(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    q = "Question: Apply the latent structure defined in the context above to identify the optimal [start, end] sequence bounds. What is the exact mathematical sum of the elements contained strictly within those optimal bounds? Give only the exact integer.\nAnswer: "
    return q, str(struct["max_sum"])


def _clrs_maxsub_probe_ood(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    ood_length = len(struct["inputs"]) * 2
    ood_array = [rng.randint(-500, -1) for _ in range(ood_length)]

    best_sum, current_sum = float("-inf"), 0
    best_l, best_h, current_l = 0, 0, 0
    for i, val in enumerate(ood_array):
        if current_sum <= 0:
            current_l = i
            current_sum = val
        else:
            current_sum += val
        if current_sum > best_sum:
            best_sum = current_sum
            best_l = current_l
            best_h = i

    q = f"Question: Apply the latent structure defined in the context above to this entirely new, adversarial Target Array: {ood_array}. Identify the 0-indexed [start, end] bounds of the optimal sequence. Answer with the list in form [start, end]. One line only.\nAnswer: "
    return q, str([best_l, best_h])


def probe_clrs_max_subarray(
    struct: Dict[str, Any], held_out: List[Any], application: str, rng: random.Random
) -> Tuple[str, str]:
    probes = {
        "forward": _clrs_maxsub_probe_forward,
        "inverse": _clrs_maxsub_probe_inverse,
        "ood": _clrs_maxsub_probe_ood,
    }
    return probes[application](struct, held_out, rng)


# ============================================================================
# TASK 3: CLRS BINARY SEARCH
# ============================================================================

CLRS_BSEARCH_DIFFICULTY = {
    1: {"length": 7, "n_fs_examples": 3},
    2: {"length": 15, "n_fs_examples": 4},
    3: {"length": 31, "n_fs_examples": 5},
}


def generate_clrs_binary_search(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    cfg = CLRS_BSEARCH_DIFFICULTY[difficulty]

    sampler, spec = _clrs_build_sampler(
        name="binary_search",
        seed=seed,
        num_samples=1,
        length=cfg["length"],
        track_max_steps=True,
    )
    sample = sampler.next(batch_size=1)

    raw_key = next(f.data[0] for f in sample.features.inputs if f.name == "key")
    raw_target = next(f.data[0] for f in sample.features.inputs if f.name == "target")

    target_array = sorted([round(x * 100) for x in raw_key.tolist()])
    target_val = round(float(raw_target) * 100)

    raw_return = next(f.data[0] for f in sample.outputs if f.name == "return")
    final_index = int(np.argmax(raw_return)) if np.any(raw_return) else -1

    raw_low_h = next(f.data for f in sample.features.hints if f.name == "low")
    raw_high_h = next(f.data for f in sample.features.hints if f.name == "high")

    trace_states = []
    last_state = None
    for low_h, high_h in zip(raw_low_h, raw_high_h):
        lo, hi = int(np.argmax(low_h[0])), int(np.argmax(high_h[0]))
        if (lo, hi) != last_state:
            trace_states.append((lo, hi))
            last_state = (lo, hi)

    return {
        "task_type": "binary_search",
        "difficulty": difficulty,
        "target_array": target_array,
        "target_val": target_val,
        "final_index": final_index,
        "trace_states": trace_states,
        "seed": seed,
    }


# --- FORMATTERS ---


def _clrs_bsearch_format_demonstration(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    cfg = CLRS_BSEARCH_DIFFICULTY[struct["difficulty"]]
    sampler, _ = _clrs_build_sampler(
        name="binary_search",
        seed=struct["seed"] + 1,
        num_samples=cfg["n_fs_examples"],
        length=cfg["length"],
        track_max_steps=False,
    )
    lines = []
    for _ in range(cfg["n_fs_examples"]):
        sample = sampler.next(batch_size=1)
        fs_key = sorted(
            [
                round(x * 100)
                for x in next(
                    f.data[0] for f in sample.features.inputs if f.name == "key"
                ).tolist()
            ]
        )
        fs_target = round(
            float(next(f.data[0] for f in sample.features.inputs if f.name == "target"))
            * 100
        )
        fs_ret = int(
            np.argmax(next(f.data[0] for f in sample.outputs if f.name == "return"))
        )
        lines.append(f"Array: {fs_key} | Target: {fs_target} | Output Index: {fs_ret}")

    lines.append(
        f"Target Array: {struct['target_array']}\nSearch Target: {struct['target_val']}"
    )
    return "\n".join(lines) + "\n\n", []


def _clrs_bsearch_format_declarative_natural_language(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    # UPGRADED: Explicitly define the Lower Bound (first occurrence) tie-breaker
    return (
        f"Rule: Find the 0-indexed position of the first occurrence of the Search Target in the sorted array by repeatedly halving the search interval.\n\nTarget Array: {struct['target_array']}\nSearch Target: {struct['target_val']}\n\n",
        [],
    )


def _clrs_bsearch_format_trace(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    trace = struct["trace_states"]
    lines = [
        f"Target Array: {struct['target_array']}\nSearch Target: {struct['target_val']}",
        "Trace (low and high indices):",
    ]
    for i, state in enumerate(trace):
        lines.append(f"Step {i}: Bounds [low: {state[0]}, high: {state[1]}]")
    return "\n".join(lines) + "\n\n", []


def _clrs_bsearch_format_formal_specification(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    math_spec = "Formal Specification:\nInitialize L = 0, R = N - 1\nLoop while L < R:\n  mid = ⌊(L + R) / 2⌋\n  if A[mid] < target: L = mid + 1\n  else: R = mid\nif A[L] == target: return L\nelse: return -1\n\n"
    data = {
        "A": struct["target_array"],
        "target": struct["target_val"],
    }  # Matched variables A and target
    return math_spec + "Data:\n```json\n" + json.dumps(data) + "\n```\n\n", []


def format_clrs_binary_search(
    struct: Dict[str, Any], fmt: str, app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    struct["current_format"] = fmt
    formatters = {
        "demonstration": _clrs_bsearch_format_demonstration,
        "declarative_natural_language": _clrs_bsearch_format_declarative_natural_language,
        "trace": _clrs_bsearch_format_trace,
        "formal_specification": _clrs_bsearch_format_formal_specification,
    }
    return formatters[fmt](struct, app, rng)


# --- PROBES ---


# UPGRADED: Rule-agnostic.
def _clrs_bsearch_probe_forward(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    q = "Question: Apply the latent structure defined in the context above to locate the Search Target within the Target Array. At what exact 0-indexed position does the interval ultimately halt and return its final pointer? Give only the exact integer.\nAnswer: "
    return q, str(struct["final_index"])


def _clrs_bsearch_probe_inverse(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    # UPGRADED: Filter out duplicate indices to prevent Unreachable Halting State paradoxes
    arr = struct["target_array"]
    valid_indices = [i for i in range(len(arr)) if i == 0 or arr[i] != arr[i - 1]]

    hypothetical_idx = rng.choice(valid_indices)
    target_val_at_idx = arr[hypothetical_idx]

    q = f"Question: Assume the latent structure defined in the context above is applied to the provided Target Array, but with a different Search Target. If the procedure ultimately halts and returns its final pointer at 0-indexed position {hypothetical_idx}, what exact integer must the new Search Target be? Give only the exact integer.\nAnswer: "
    return q, str(target_val_at_idx)


def _clrs_bsearch_probe_ood(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    ood_array = sorted(
        [rng.randint(-1000, 1000) for _ in range(len(struct["target_array"]) * 3)]
    )
    ood_target = ood_array[0] - 100

    q = f"Question: Apply the latent structure defined in the context above using this entirely new, adversarial Target Array: {ood_array}, and a new Search Target: {ood_target}. At what exact 0-indexed position does the interval ultimately halt and return its final pointer? If it does not exist, return -1. Give only the exact integer.\nAnswer: "
    return q, "-1"


def probe_clrs_binary_search(
    struct: Dict[str, Any], held_out: List[Any], application: str, rng: random.Random
) -> Tuple[str, str]:
    probes = {
        "forward": _clrs_bsearch_probe_forward,
        "inverse": _clrs_bsearch_probe_inverse,
        "ood": _clrs_bsearch_probe_ood,
    }
    return probes[application](struct, held_out, rng)


# ============================================================================
# TASK 4: CLRS NAIVE STRING MATCHER
# ============================================================================

CLRS_NSTRING_DIFFICULTY = {
    1: {"text_len": 10, "pat_len": 2, "n_fs_examples": 3},
    2: {"text_len": 16, "pat_len": 3, "n_fs_examples": 4},
    3: {"text_len": 24, "pat_len": 4, "n_fs_examples": 5},
}


def generate_clrs_naive_string_matcher(
    seed: int = 42, difficulty: int = 1
) -> Dict[str, Any]:
    rng = random.Random(seed)
    cfg = CLRS_NSTRING_DIFFICULTY[difficulty]

    text = [rng.randint(0, 5) for _ in range(cfg["text_len"])]
    pattern = [rng.randint(0, 5) for _ in range(cfg["pat_len"])]

    implant_idx = rng.randint(0, len(text) - len(pattern))
    text[implant_idx : implant_idx + len(pattern)] = pattern

    matches = []
    for i in range(len(text) - len(pattern) + 1):
        if text[i : i + len(pattern)] == pattern:
            matches.append(i)

    trace_states = []
    pat_len = len(pattern)
    for i in range(len(text) - pat_len + 1):
        trace_states.append({"shift": i, "window": text[i : i + pat_len]})

    return {
        "task_type": "naive_string_matcher",
        "difficulty": difficulty,
        "text_array": text,
        "pattern_array": pattern,
        "matches": matches,
        "trace_states": trace_states,
        "seed": seed,
    }


# --- FORMATTERS ---


def _clrs_nstring_format_demonstration(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    lines = []
    for _ in range(CLRS_NSTRING_DIFFICULTY[struct["difficulty"]]["n_fs_examples"]):
        t = [rng.randint(0, 3) for _ in range(8)]
        p = [rng.randint(0, 3) for _ in range(2)]
        m = [i for i in range(len(t) - len(p) + 1) if t[i : i + len(p)] == p]
        lines.append(f"Text: {t} | Pattern: {p} | Output Matches: {m}")

    lines.append(
        f"Target Text: {struct['text_array']}\nTarget Pattern: {struct['pattern_array']}"
    )
    return "\n".join(lines) + "\n\n", []


def _clrs_nstring_format_declarative_natural_language(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    return (
        f"Rule: Scan the Target Text from left to right to find all 0-indexed starting positions where the Target Pattern occurs exactly.\n\nTarget Text: {struct['text_array']}\nTarget Pattern: {struct['pattern_array']}\n\n",
        [],
    )


def _clrs_nstring_format_trace(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    trace = struct["trace_states"]
    lines = [
        f"Target Text: {struct['text_array']}\nTarget Pattern: {struct['pattern_array']}",
        "Trace (Window slice evaluated at each shift):",
    ]
    for i, state in enumerate(trace):
        lines.append(
            f"Step {i}: Shift {state['shift']} extracts window {state['window']}"
        )
    return "\n".join(lines) + "\n\n", []


def _clrs_nstring_format_formal_specification(
    struct: Dict[str, Any], app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    # Replaced len() with | | cardinality notation
    math_spec = "Formal Specification:\nM = { s ∈ [0, |T| - |P|] | ∀ j ∈ [0, |P| - 1], T[s+j] == P[j] }\n\n"
    data = {
        "T": struct["text_array"],
        "P": struct["pattern_array"],
    }  # Matched variables T and P
    return math_spec + "Data:\n```json\n" + json.dumps(data) + "\n```\n\n", []


def format_clrs_naive_string_matcher(
    struct: Dict[str, Any], fmt: str, app: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    struct["current_format"] = fmt
    formatters = {
        "demonstration": _clrs_nstring_format_demonstration,
        "declarative_natural_language": _clrs_nstring_format_declarative_natural_language,
        "trace": _clrs_nstring_format_trace,
        "formal_specification": _clrs_nstring_format_formal_specification,
    }
    return formatters[fmt](struct, app, rng)


# --- PROBES ---


# UPGRADED: Rule-agnostic.
def _clrs_nstring_probe_forward(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    q = "Question: Apply the latent structure defined in the context above to scan the Target Text. List all 0-indexed starting positions where a valid match is identified. Answer with the list in form [index1, index2, ...]. If none, output []. One line only.\nAnswer: "
    return q, str(struct["matches"])


def _clrs_nstring_probe_inverse(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    text = struct["text_array"]
    pat = struct["pattern_array"]
    mismatches = [
        i for i in range(len(text) - len(pat) + 1) if i not in struct["matches"]
    ]

    if not mismatches:
        return (
            "Question: If we extract a slice of the Target Text equal in length to the Target Pattern starting at index 0, what is that exact slice? Answer in form [item1, item2, ...].\nAnswer: ",
            str(text[: len(pat)]),
        )

    shift = rng.choice(mismatches)
    window = text[shift : shift + len(pat)]

    mismatch_idx = 0
    for i, (t_char, p_char) in enumerate(zip(window, pat)):
        if t_char != p_char:
            mismatch_idx = i
            break

    q = f"Question: Apply the latent structure defined in the context above. The specific starting 0-indexed position {shift} within the Target Text does NOT result in a valid match. If you align the Target Pattern starting at this position, at what exact relative 0-indexed position (the smallest index j within the Target Pattern itself) does a mismatch occur? Give only the exact integer.\nAnswer: "
    return q, str(mismatch_idx)


def _clrs_nstring_probe_ood(
    struct: Dict[str, Any], held_out: List[Any], rng: random.Random
) -> Tuple[str, str]:
    char = rng.randint(7, 9)
    ood_text = [char] * 50
    ood_pat = [char] * 5
    ood_matches = [i for i in range(len(ood_text) - len(ood_pat) + 1)]

    q = f"Question: Apply the latent structure defined in the context above using this entirely new, adversarial Target Text: {ood_text}, and a new Target Pattern: {ood_pat}. List all 0-indexed starting positions where a valid match is identified. Answer with the list in form [index1, index2, ...]. One line only.\nAnswer: "
    return q, str(ood_matches)


def probe_clrs_naive_string_matcher(
    struct: Dict[str, Any], held_out: List[Any], application: str, rng: random.Random
) -> Tuple[str, str]:
    probes = {
        "forward": _clrs_nstring_probe_forward,
        "inverse": _clrs_nstring_probe_inverse,
        "ood": _clrs_nstring_probe_ood,
    }
    return probes[application](struct, held_out, rng)


# ============================================================================
# CARTESIAN DATASET GENERATOR
# ============================================================================


def build_clift_cartesian_dataset(
    num_instances: int = 100, difficulty: int = 3
) -> List[Dict[str, Any]]:
    """
    Generates a rigorous Cartesian dataset ensuring the exact same latent
    task instances (arrays/world states) are cross-evaluated across all
    formats and applications.
    """
    formats = [
        "demonstration",
        "declarative_natural_language",
        "trace",
        "formal_specification",
    ]
    applications = ["forward", "inverse", "ood"]

    tasks = {
        "insertion_sort": (
            generate_clrs_insertion_sort,
            format_clrs_insertion_sort,
            probe_clrs_insertion_sort,
        ),
        "max_subarray": (
            generate_clrs_max_subarray,
            format_clrs_max_subarray,
            probe_clrs_max_subarray,
        ),
        "binary_search": (
            generate_clrs_binary_search,
            format_clrs_binary_search,
            probe_clrs_binary_search,
        ),
        "naive_string_matcher": (
            generate_clrs_naive_string_matcher,
            format_clrs_naive_string_matcher,
            probe_clrs_naive_string_matcher,
        ),
    }

    dataset = []
    global_id = 0
    rng = random.Random(42)

    for task_name, (gen_fn, fmt_fn, probe_fn) in tasks.items():
        # Step 1: Generate stable latent world states
        latent_instances = [
            gen_fn(seed=s, difficulty=difficulty) for s in range(num_instances)
        ]

        # Step 2: Cartesian Expansion
        for struct in latent_instances:
            for fmt in formats:
                for app in applications:
                    # Create an isolated deep copy to prevent state leakage
                    isolated_struct = copy.deepcopy(struct)

                    # Apply formatting (Context) to the isolated copy
                    prompt_context, _ = fmt_fn(isolated_struct, fmt, app, rng)

                    # Apply Application (Interrogation Vector) to the isolated copy
                    question, target = probe_fn(isolated_struct, [], app, rng)

                    dataset.append(
                        {
                            "instance_id": global_id,
                            "task": task_name,
                            "format": fmt,
                            "application": app,
                            "difficulty": difficulty,
                            "seed": isolated_struct["seed"],
                            "prompt": prompt_context + question,
                            "target": target,
                            "latent_structure": isolated_struct,
                        }
                    )
                    global_id += 1

    return dataset


__all__ = [
    "CLRS_ISORT_DIFFICULTY",
    "generate_clrs_insertion_sort",
    "format_clrs_insertion_sort",
    "probe_clrs_insertion_sort",
    "CLRS_MAXSUB_DIFFICULTY",
    "generate_clrs_max_subarray",
    "format_clrs_max_subarray",
    "probe_clrs_max_subarray",
    "CLRS_BSEARCH_DIFFICULTY",
    "generate_clrs_binary_search",
    "format_clrs_binary_search",
    "probe_clrs_binary_search",
    "CLRS_NSTRING_DIFFICULTY",
    "generate_clrs_naive_string_matcher",
    "format_clrs_naive_string_matcher",
    "probe_clrs_naive_string_matcher",
    "build_clift_cartesian_dataset",
]
