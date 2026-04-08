import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Word pools
# ---------------------------------------------------------------------------

ROOM_WORDS = [
    "kitchen",
    "bedroom",
    "bathroom",
    "garden",
    "attic",
    "cellar",
    "library",
    "parlor",
    "hallway",
    "study",
    "pantry",
    "nursery",
    "balcony",
    "dungeon",
    "chamber",
    "gallery",
    "terrace",
    "cottage",
    "tower",
    "chapel",
    "armory",
    "vault",
    "alcove",
    "foyer",
    "cloister",
    "lodge",
    "grotto",
    "rotunda",
]

ITEM_WORDS = [
    "ant",
    "bear",
    "cat",
    "dog",
    "eagle",
    "fox",
    "goat",
    "horse",
    "iguana",
    "jaguar",
    "koala",
    "lion",
    "moose",
    "newt",
    "owl",
    "panda",
]

DOMAIN_WORDS = [
    "ruby",
    "topaz",
    "jade",
    "opal",
    "onyx",
    "pearl",
    "amber",
    "coral",
    "ivory",
    "slate",
    "quartz",
    "bronze",
]

VARIABLE_WORDS = [
    "pulse",
    "flux",
    "drift",
    "surge",
    "bloom",
    "glow",
    "haze",
    "spark",
    "crest",
    "loop",
    "tide",
    "veil",
]

ACTION_WORDS = ["push", "pull", "turn", "press", "slide"]
OBSERVATION_WORDS = ["beep", "flash", "buzz", "click", "hum"]

# ---------------------------------------------------------------------------
# Canonical axis labels
# ---------------------------------------------------------------------------

TASKS: List[str] = [
    "lookup_table",
    "arithmetic_rule",
    "conditional_rule",
    "insertion_sort",
    "max_subarray",
    "binary_search",
    "naive_string_matcher",
    "spatial_translation",
    "affine_dynamics_2d",
    "register_machine_2d",
]

FAMILIES: Dict[str, List[str]] = {
    "functional_mappings": ["lookup_table", "arithmetic_rule", "conditional_rule"],
    "algorithmic": [
        "insertion_sort",
        "max_subarray",
        "binary_search",
        "naive_string_matcher",
    ],
    "spatial": ["spatial_translation"],
    "dynamic_structures": ["affine_dynamics_2d", "register_machine_2d"],
}

FORMATS: List[str] = [
    "demonstration",
    "natural_language",
    "trace",
    "formal_spec",
]

# Per-task application axes (affine_dynamics_2d uses OOD-suffixed probe names).
APPLICATIONS_STANDARD: List[str] = [
    "forward",
    "inverse",
    "articulation",
    "ood",
    "planning",
    "structural",
]
APPLICATIONS_AFFINE_2D: List[str] = [
    "forward",
    "inverse",
    "ood@forward",
    "ood@traj",
    "ood@inverse",
]
APPLICATIONS_REGISTER_2D: List[str] = [
    "forward",
    "inverse",
    "ood@forward",
    "ood@traj",
    "ood@inverse",
]
APPLICATIONS: List[str] = [
    *APPLICATIONS_STANDARD,
    "ood@forward",
    "ood@traj",
    "ood@inverse",
]
DIFFICULTIES: List[int] = [1, 2, 3]


def applications_for_task(task: str) -> List[str]:
    """Applications generated for *task* (subset of APPLICATIONS)."""
    if task == "affine_dynamics_2d":
        return list(APPLICATIONS_AFFINE_2D)
    if task == "register_machine_2d":
        return list(APPLICATIONS_REGISTER_2D)
    return list(APPLICATIONS_STANDARD)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class CLIFTInstance:
    """A single evaluation instance in the CLIFT benchmark (alias: LIFTInstance)."""

    task: str
    format: str
    application: str
    prompt: str
    target: str
    difficulty: int = 1
    latent_structure: Dict[str, Any] = field(default_factory=dict)
    instance_id: int = 0
    instruct: bool = False
    messages: Optional[List[Dict[str, str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "instance_id": self.instance_id,
            "task": self.task,
            "format": self.format,
            "application": self.application,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "target": self.target,
            "latent_structure": self.latent_structure,
            "instruct": self.instruct,
        }
        if self.messages is not None:
            d["messages"] = self.messages
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def mod_inverse(a: int, p: int) -> int:
    """Modular multiplicative inverse via Fermat's little theorem.

    *p* must be prime and not divide *a* (standard finite-field preconditions).
    """
    return pow(a, p - 2, p)


def affine_period(a: int, b: int, p: int) -> int:
    """Period of the affine map x -> ax + b on Z_p."""
    if a == 1:
        return 1 if b == 0 else p
    for k in range(1, p):
        if pow(a, k, p) == 1:
            return k
    return p - 1


def permutation_cycle_info(mapping: Dict[str, str]) -> Tuple[int, int]:
    """Return (number of fixed points, longest cycle length) of a permutation.

    *mapping* must be closed under iteration (every value is a key), as for a
    permutation on a finite set; otherwise lookup may raise ``KeyError``.
    """
    visited: set[str] = set()
    fixed = 0
    longest = 0
    for start in mapping:
        if start in visited:
            continue
        cycle_len = 0
        current = start
        while current not in visited:
            visited.add(current)
            current = mapping[current]
            cycle_len += 1
        if cycle_len == 1 and mapping[start] == start:
            fixed += 1
        longest = max(longest, cycle_len)
    return fixed, longest


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


def export_jsonl(instances: List[CLIFTInstance], path: str | Path) -> None:
    """Export LIFT instances to a HuggingFace-compatible JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst.to_dict()) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load LIFT instances from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records
