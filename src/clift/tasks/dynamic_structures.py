import itertools
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Shared formatting / probe instructions (affine + register probes)
# ---------------------------------------------------------------------------


def _fmt_state(s: Tuple[int, int]) -> str:
    return f"({s[0]}, {s[1]})"


def _fmt_trajectory(traj: List[Tuple[int, int]]) -> str:
    return " -> ".join(_fmt_state(s) for s in traj)


# Probe prompts: how the model must format answers (must match _fmt_state / _fmt_trajectory).
_PROBE_SINGLE_STATE_REPLY_SPEC = (
    "Reply on a single line. Put only the state, in the form (x, y): "
    "parentheses around two integers, with a comma and one space between them. "
    "Do not add a label, explanation, markdown, or a second line."
)
_PROBE_TRAJECTORY_REPLY_SPEC = (
    "Reply on a single line. List every state from step 0 through step T in order. "
    "Write each state as (x, y) with a comma and one space after the comma. "
    "Separate consecutive states with ' -> ' (space, hyphen, greater-than, space), "
    "like this: (0, 1) -> (2, 0) -> (3, 4). "
    "Do not prefix steps with numbers or add any other text."
)
_PROBE_STEP_INDEX_REPLY_SPEC = (
    "Reply on a single line with only one non-negative integer: the step index. "
    "Do not add a label, explanation, markdown, or a second line."
)


# ============================================================================
# 2D AFFINE DYNAMICS TASK
# ============================================================================


class AffineDynamics2DTask:
    """2D affine linear dynamical system over Z_p with uniquely determining demos."""

    UNIQUE_USE_POOL_MIN_P = 12
    K_DEMOS = 20

    # Must match keys in probe(); one held-out initial state each.
    PROBE_APPLICATIONS: Tuple[str, ...] = (
        "forward",
        "inverse",
        "ood@forward",
        "ood@traj",
        "ood@inverse",
    )

    DIFFICULTY = {
        1: {
            "p": 5,
            "ctx_horizon": (3, 5),
            "iid_horizon": (5, 8),
            "ood_horizon": (10, 16),
        },
        2: {
            "p": 7,
            "ctx_horizon": (4, 7),
            "iid_horizon": (8, 14),
            "ood_horizon": (16, 28),
        },
        3: {
            "p_choices": [11, 13],
            "ctx_horizon": (5, 8),
            "iid_horizon": (14, 22),
            "ood_horizon": (28, 45),
        },
    }

    @staticmethod
    def _grid_states(p: int) -> List[Tuple[int, int]]:
        return list(itertools.product(range(p), range(p)))

    @classmethod
    def _partition_probe_inits(
        cls, rng: random.Random, p: int
    ) -> Tuple[Dict[str, Tuple[int, int]], frozenset, List[Tuple[int, int]]]:
        """Reserve one distinct state per probe application; demos use the rest."""
        n_app = len(cls.PROBE_APPLICATIONS)
        n_states = p * p
        if n_states <= n_app:
            raise ValueError(
                f"affine_dynamics_2d: require p^2 > len(PROBE_APPLICATIONS); "
                f"got p={p}, n_states={n_states}, n_app={n_app}"
            )
        all_states = cls._grid_states(p)
        held_list = rng.sample(all_states, n_app)
        held_set = frozenset(held_list)
        probe_init_by_application = dict(zip(cls.PROBE_APPLICATIONS, held_list))
        demo_states = [s for s in all_states if s not in held_set]
        return probe_init_by_application, held_set, demo_states

    @classmethod
    def _chosen_demo_pairs(
        cls,
        rng: random.Random,
        demo_states: Sequence[Tuple[int, int]],
        h_lo: int,
        h_hi: int,
    ) -> List[Tuple[Tuple[int, int], int]]:
        pool = [(s, T) for s in demo_states for T in range(h_lo, h_hi + 1)]
        if len(pool) <= cls.K_DEMOS:
            pairs = list(pool)
            rng.shuffle(pairs)
        else:
            pairs = rng.sample(pool, cls.K_DEMOS)
        return pairs

    @staticmethod
    def _step(coeffs: tuple, state: Tuple[int, int], p: int) -> Tuple[int, int]:
        """Single step: (x, y) -> (a*x + b*y + c, d*x + e*y + f) mod p."""
        a, b, c, d, e, f = coeffs
        x, y = state
        return ((a * x + b * y + c) % p, (d * x + e * y + f) % p)

    @staticmethod
    def _rollout(
        coeffs: tuple, state: Tuple[int, int], T: int, p: int
    ) -> List[Tuple[int, int]]:
        """Full trajectory [s_0, s_1, ..., s_T] including initial state."""
        traj = [state]
        for _ in range(T):
            state = AffineDynamics2DTask._step(coeffs, state, p)
            traj.append(state)
        return traj

    @staticmethod
    def _final_state(
        coeffs: tuple, state: Tuple[int, int], T: int, p: int
    ) -> Tuple[int, int]:
        """State after T steps."""
        for _ in range(T):
            state = AffineDynamics2DTask._step(coeffs, state, p)
        return state

    @staticmethod
    def _consistent_demo(candidate: tuple, demos: List[Dict], p: int) -> bool:
        """Check if candidate matches demonstration endpoints."""
        for demo in demos:
            if (
                AffineDynamics2DTask._final_state(
                    candidate, demo["initial"], demo["horizon"], p
                )
                != demo["final"]
            ):
                return False
        return True

    @staticmethod
    def _consistent_trace(candidate: tuple, demos: List[Dict], p: int) -> bool:
        """Check if candidate matches full trajectories."""
        for demo in demos:
            traj = demo["trajectory"]
            if (
                AffineDynamics2DTask._rollout(
                    candidate, demo["initial"], len(traj) - 1, p
                )
                != traj
            ):
                return False
        return True

    @staticmethod
    def _demos_for_unique_check(demos: List[Dict]) -> List[Dict]:
        """Order demos for faster rejection in _consistent_demo (cheap checks first)."""
        return sorted(demos, key=lambda d: (d["horizon"], d["initial"]))

    @classmethod
    def _system_unique_from_demos(cls, p: int, demos: List[Dict]) -> Tuple[bool, int]:
        """Check if exactly one system in Z_p^6 matches the demo endpoints.

        Returns (is_unique, num_consistent).
        """
        demos_s = cls._demos_for_unique_check(demos)
        total = 0

        if p < cls.UNIQUE_USE_POOL_MIN_P:
            for a_val in range(p):
                total += _affine_pool_check_slice((a_val, p, demos_s))
                if total > 1:
                    return False, total
            return total == 1, total

        args = [(a_val, p, demos_s) for a_val in range(p)]
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_affine_pool_check_slice, a): a for a in args}
            for fut in as_completed(futures):
                total += fut.result()
                if total > 1:
                    for f in futures:
                        f.cancel()
                    break
        return total == 1, total

    def generate(self, seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
        """Generate a 2D affine dynamical system with uniquely determining demos."""
        rng = random.Random(seed)
        cfg = self.DIFFICULTY[difficulty]

        if difficulty == 3:
            p = rng.choice(cfg["p_choices"])
        else:
            p = cfg["p"]

        h_lo, h_hi = cfg["ctx_horizon"]
        probe_init_by_application, _held_set, demo_states = self._partition_probe_inits(
            rng, p
        )

        coeffs = None
        demos = None
        n_consistent = 0
        is_unique = False

        for _attempt in range(500):
            coeffs = tuple(rng.randint(0, p - 1) for _ in range(6))
            demo_pairs = self._chosen_demo_pairs(rng, demo_states, h_lo, h_hi)

            demos = []
            for init, T in demo_pairs:
                traj = self._rollout(coeffs, init, T, p)
                demos.append(
                    {
                        "initial": init,
                        "horizon": T,
                        "final": traj[-1],
                        "trajectory": traj,
                    }
                )

            is_unique, n_consistent = self._system_unique_from_demos(p, demos)
            if is_unique:
                break

        return {
            "task_type": "affine_dynamics_2d",
            "difficulty": difficulty,
            "p": p,
            "coeffs": list(coeffs),
            "demonstrations": [
                {
                    "initial": list(d["initial"]),
                    "horizon": d["horizon"],
                    "final": list(d["final"]),
                    "trajectory": [list(s) for s in d["trajectory"]],
                }
                for d in demos
            ],
            "probe_init_by_application": {
                k: [v[0], v[1]] for k, v in sorted(probe_init_by_application.items())
            },
            "num_consistent_candidates": n_consistent,
            "context_valid": is_unique,
            "answer_unique": is_unique,
            "_coeffs": coeffs,
            "_demos": demos,
            "_probe_init_by_application": probe_init_by_application,
        }

    def _fmt_few_shot(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        """Demonstration format: endpoint-only solved examples."""
        demos = struct["_demos"]
        lines = []
        for demo in demos:
            x0, y0 = demo["initial"]
            xT, yT = demo["final"]
            T = demo["horizon"]
            lines.append(
                f"Initial state: ({x0}, {y0}), Steps: {T}, Final state: ({xT}, {yT})"
            )
        ctx = (
            "The following examples show a 2D dynamical system over integers.\n"
            "Each example gives an initial state, the number of steps, "
            "and the resulting final state.\n\n" + "\n".join(lines) + "\n\n"
        )
        return ctx, [{"format": "demonstration", "demos": demos}]

    def _fmt_step_by_step(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        """Trace format: full trajectories."""
        demos = struct["_demos"]
        blocks = []
        for i, demo in enumerate(demos, 1):
            traj_str = " -> ".join(f"({x}, {y})" for x, y in demo["trajectory"])
            blocks.append(f"Trajectory {i}: {traj_str}")
        ctx = (
            "The following trajectories show the step-by-step evolution "
            "of a 2D dynamical system.\n\n" + "\n".join(blocks) + "\n\n"
        )
        return ctx, [{"format": "trace", "demos": demos}]

    def _fmt_structured(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        """Formal specification: explicit equations."""
        p = struct["p"]
        a, b, c, d, e, f = struct["_coeffs"]
        ctx = (
            f"A 2D dynamical system over integers modulo {p}:\n"
            f"  x_{{t+1}} = {a}*x_t + {b}*y_t + {c}  (mod {p})\n"
            f"  y_{{t+1}} = {d}*x_t + {e}*y_t + {f}  (mod {p})\n\n"
        )
        return ctx, [{"format": "formal_spec"}]

    def _fmt_natural_language(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        """High-level description: iterated affine map on (Z/pZ)^2 with concrete parameters."""
        p = struct["p"]
        a, b, c, d, e, f = struct["_coeffs"]
        ctx = (
            f"This system evolves step by step on ordered pairs of integers modulo {p}. "
            f"At each step, the current pair (x, y) is used to generate a new pair: "
            f"the new x-value comes from a fixed combination of x and y plus an offset, "
            f"and the new y-value is formed the same way but with its own coefficients and offset. "
            f"Afterward, both values are reduced modulo {p}. "
            f"Here, the x-update uses coefficients {a} and {b} with offset {c}, "
            f"while the y-update uses coefficients {d} and {e} with offset {f}.\n\n"
        )
        return ctx, [{"format": "declarative_nl"}]

    def format(
        self, struct: Dict[str, Any], fmt: str, rng: random.Random
    ) -> Tuple[str, List[Any]]:
        """Format a 2D affine dynamics instance."""
        if fmt == "demonstration":
            return self._fmt_few_shot(struct, rng)
        if fmt == "natural_language":
            return self._fmt_natural_language(struct, rng)
        if fmt == "trace":
            return self._fmt_step_by_step(struct, rng)
        if fmt == "formal_spec":
            return self._fmt_structured(struct, rng)
        raise KeyError(fmt)

    @staticmethod
    def _first_occurrence_step(
        traj: List[Tuple[int, int]], target: Tuple[int, int]
    ) -> int:
        """Smallest t with traj[t] == target (target is always taken from traj)."""
        for t, s in enumerate(traj):
            if s == target:
                return t
        raise RuntimeError("target state not in trajectory")

    def _probe_forward(
        self, struct: Dict[str, Any], held_out: List[Any], rng: random.Random
    ) -> Tuple[str, str]:
        """IID final_state: predict (x_T, y_T)."""
        p = struct["p"]
        coeffs = struct["_coeffs"]
        init = struct["_probe_init_by_application"]["forward"]
        cfg = self.DIFFICULTY[struct["difficulty"]]
        lo, hi = cfg["iid_horizon"]
        T = rng.randint(max(lo, 2), hi)

        final = self._final_state(coeffs, init, T, p)
        q = (
            f"Question: Starting from state {_fmt_state(init)}, "
            f"what is the state after {T} steps?\n\n"
            f"{_PROBE_SINGLE_STATE_REPLY_SPEC}\n\n"
            f"Answer: "
        )
        return q, _fmt_state(final)

    def _probe_inverse(
        self, struct: Dict[str, Any], held_out: List[Any], rng: random.Random
    ) -> Tuple[str, str]:
        """IID first_occurrence: step index where a given state first appears (steps 0..T)."""
        p = struct["p"]
        coeffs = struct["_coeffs"]
        init = struct["_probe_init_by_application"]["inverse"]
        cfg = self.DIFFICULTY[struct["difficulty"]]
        lo, hi = cfg["iid_horizon"]
        T = rng.randint(max(lo, 2), hi)

        traj = self._rollout(coeffs, init, T, p)
        k = rng.randint(1, T)
        target = traj[k]
        t_first = self._first_occurrence_step(traj, target)

        q = (
            f"Question: Starting from state {_fmt_state(init)}, the system evolves for {T} steps. "
            f"Step 0 is the initial state; each further step applies the same transition once, "
            f"so the states at steps 0 through {T} are visited in order. "
            f"At which step index does the state {_fmt_state(target)} "
            f"appear for the first time among steps 0 through {T}?\n\n"
            f"{_PROBE_STEP_INDEX_REPLY_SPEC}\n\n"
            f"Answer: "
        )
        return q, str(t_first)

    def _probe_ood(
        self, struct: Dict[str, Any], held_out: List[Any], rng: random.Random
    ) -> Tuple[str, str]:
        """OOD final_state: longer horizon."""
        p = struct["p"]
        coeffs = struct["_coeffs"]
        init = struct["_probe_init_by_application"]["ood@forward"]
        cfg = self.DIFFICULTY[struct["difficulty"]]
        lo, hi = cfg["ood_horizon"]
        T = rng.randint(max(lo, 2), hi)

        final = self._final_state(coeffs, init, T, p)
        q = (
            f"Question: Starting from state {_fmt_state(init)}, "
            f"what is the state after {T} steps?\n\n"
            f"{_PROBE_SINGLE_STATE_REPLY_SPEC}\n\n"
            f"Answer: "
        )
        return q, _fmt_state(final)

    def _probe_planning(
        self, struct: Dict[str, Any], held_out: List[Any], rng: random.Random
    ) -> Tuple[str, str]:
        """OOD trajectory: full trajectory with longer horizon."""
        p = struct["p"]
        coeffs = struct["_coeffs"]
        init = struct["_probe_init_by_application"]["ood@traj"]
        cfg = self.DIFFICULTY[struct["difficulty"]]
        lo, hi = cfg["ood_horizon"]
        T = rng.randint(max(lo, 2), hi)

        traj = self._rollout(coeffs, init, T, p)
        q = (
            f"Question: Starting from state {_fmt_state(init)}, "
            f"list every state from step 0 to step {T}.\n\n"
            f"{_PROBE_TRAJECTORY_REPLY_SPEC}\n\n"
            f"Answer: "
        )
        return q, _fmt_trajectory(traj)

    def _probe_structural(
        self, struct: Dict[str, Any], held_out: List[Any], rng: random.Random
    ) -> Tuple[str, str]:
        """OOD first_occurrence: same as inverse with longer horizon."""
        p = struct["p"]
        coeffs = struct["_coeffs"]
        init = struct["_probe_init_by_application"]["ood@inverse"]
        cfg = self.DIFFICULTY[struct["difficulty"]]
        lo, hi = cfg["ood_horizon"]
        T = rng.randint(max(lo, 2), hi)

        traj = self._rollout(coeffs, init, T, p)
        k = rng.randint(1, T)
        target = traj[k]
        t_first = self._first_occurrence_step(traj, target)

        q = (
            f"Question: Starting from state {_fmt_state(init)}, the system evolves for {T} steps. "
            f"Step 0 is the initial state; each further step applies the same transition once, "
            f"so the states at steps 0 through {T} are visited in order. "
            f"At which step index does the state {_fmt_state(target)} "
            f"appear for the first time among steps 0 through {T}?\n\n"
            f"{_PROBE_STEP_INDEX_REPLY_SPEC}\n\n"
            f"Answer: "
        )
        return q, str(t_first)

    def probe(
        self,
        struct: Dict[str, Any],
        held_out: List[Any],
        application: str,
        rng: random.Random,
    ) -> Tuple[str, str]:
        """Generate a probe question for the 2D affine dynamics task."""
        probes = {
            "forward": self._probe_forward,
            "inverse": self._probe_inverse,
            "ood@forward": self._probe_ood,
            "ood@traj": self._probe_planning,
            "ood@inverse": self._probe_structural,
        }
        return probes[application](struct, held_out, rng)


def _affine_pool_check_slice(args: Tuple) -> int:
    """Count consistent systems with a fixed first coefficient, up to 2 (picklable pool worker)."""
    a_val, p, demos = args
    count = 0
    for rest in itertools.product(range(p), repeat=5):
        if AffineDynamics2DTask._consistent_demo((a_val,) + rest, demos, p):
            count += 1
            if count > 1:
                return count
    return count


_affine_task = AffineDynamics2DTask()

K_DEMOS = AffineDynamics2DTask.K_DEMOS
AFFINE2D_PROBE_APPLICATIONS = AffineDynamics2DTask.PROBE_APPLICATIONS
AFFINE2D_DIFFICULTY = AffineDynamics2DTask.DIFFICULTY


def generate_affine_dynamics_2d(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    return _affine_task.generate(seed, difficulty)


def format_affine_dynamics_2d(
    struct: Dict[str, Any], fmt: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    return _affine_task.format(struct, fmt, rng)


def probe_affine_dynamics_2d(
    struct: Dict[str, Any],
    held_out: List[Any],
    application: str,
    rng: random.Random,
) -> Tuple[str, str]:
    return _affine_task.probe(struct, held_out, application, rng)


# ============================================================================
# 2-REGISTER INVERTIBLE MACHINE (P(6,4) = 360 hypothesis class)
# ============================================================================


class RegisterMachine2DTask:
    """Two registers over Z_p with four actions mapped to six invertible instructions."""

    INSTRUCTION_NAMES: Tuple[str, ...] = (
        "INC1",
        "INC2",
        "ADD12",
        "ADD21",
        "SWAP",
        "NEG1",
    )
    ACTION_LABELS: Tuple[str, ...] = ("A0", "A1", "A2", "A3")

    MACHINES: Tuple[Tuple[int, int, int, int], ...] = tuple(
        itertools.permutations(range(6), 4)
    )

    K_REGISTER_DEMOS = 20

    DIFFICULTY: Dict[int, Dict[str, Any]] = {
        1: {
            "p": 5,
            "ctx_horizon": (3, 5),
            "iid_horizon": (5, 8),
            "ood_horizon": (10, 16),
            "min_complex_per_seq": 0,
        },
        2: {
            "p": 7,
            "ctx_horizon": (4, 7),
            "iid_horizon": (8, 14),
            "ood_horizon": (16, 28),
            "min_complex_per_seq": 1,
        },
        3: {
            "p": 11,
            "ctx_horizon": (5, 8),
            "iid_horizon": (14, 24),
            "ood_horizon": (28, 45),
            "min_complex_per_seq": 2,
        },
    }

    _STEP_VERBAL: Tuple[str, ...] = (
        "Increment first component by 1 (modulo p); leave second component unchanged.",
        "Increment second component by 1 (modulo p); leave first component unchanged.",
        "Replace first component with first component plus second component (modulo p); leave second component unchanged.",
        "Replace second component with second component plus first component (modulo p); leave first component unchanged.",
        "Swap first and second components.",
        "Replace first component with −first component (modulo p); leave second component unchanged.",
    )

    _STEP_FORMAL: Tuple[str, ...] = (
        "(x, y) -> (x + 1, y)",
        "(x, y) -> (x, y + 1)",
        "(x, y) -> (x + y, y)",
        "(x, y) -> (x, y + x)",
        "(x, y) -> (y, x)",
        "(x, y) -> (-x, y)",
    )

    @staticmethod
    def _apply_instruction_forward(
        inst: int, state: Tuple[int, int], p: int
    ) -> Tuple[int, int]:
        r1, r2 = state
        if inst == 0:  # INC1
            return ((r1 + 1) % p, r2)
        if inst == 1:  # INC2
            return (r1, (r2 + 1) % p)
        if inst == 2:  # ADD12
            return ((r1 + r2) % p, r2)
        if inst == 3:  # ADD21
            return (r1, (r2 + r1) % p)
        if inst == 4:  # SWAP
            return (r2, r1)
        if inst == 5:  # NEG1
            return ((-r1) % p, r2)
        raise ValueError(f"unknown instruction index {inst}")

    @staticmethod
    def _apply_instruction_inverse(
        inst: int, state: Tuple[int, int], p: int
    ) -> Tuple[int, int]:
        r1, r2 = state
        if inst == 0:
            return ((r1 - 1) % p, r2)
        if inst == 1:
            return (r1, (r2 - 1) % p)
        if inst == 2:
            return ((r1 - r2) % p, r2)
        if inst == 3:
            return (r1, (r2 - r1) % p)
        if inst == 4:
            return (r2, r1)
        if inst == 5:
            return ((-r1) % p, r2)
        raise ValueError(f"unknown instruction index {inst}")

    @classmethod
    def step(
        cls,
        machine: Sequence[int],
        state: Tuple[int, int],
        action: int,
        p: int,
    ) -> Tuple[int, int]:
        inst = machine[action]
        return cls._apply_instruction_forward(inst, state, p)

    @classmethod
    def rollout(
        cls,
        machine: Sequence[int],
        initial_state: Tuple[int, int],
        action_sequence: Sequence[int],
        p: int,
    ) -> List[Tuple[int, int]]:
        traj = [initial_state]
        s = initial_state
        for a in action_sequence:
            s = cls.step(machine, s, int(a), p)
            traj.append(s)
        return traj

    @classmethod
    def final_state(
        cls,
        machine: Sequence[int],
        initial_state: Tuple[int, int],
        action_sequence: Sequence[int],
        p: int,
    ) -> Tuple[int, int]:
        s = initial_state
        for a in action_sequence:
            s = cls.step(machine, s, int(a), p)
        return s

    @classmethod
    def inverse_initial_state(
        cls,
        machine: Sequence[int],
        final_state: Tuple[int, int],
        action_sequence: Sequence[int],
        p: int,
    ) -> Tuple[int, int]:
        s = final_state
        for a in reversed(list(action_sequence)):
            inst = machine[a]
            s = cls._apply_instruction_inverse(inst, s, p)
        return s

    @classmethod
    def _make_candidate(
        cls,
        rng: random.Random,
        machine: Sequence[int],
        p: int,
        lo: int,
        hi: int,
        min_complex: int,
    ) -> Dict[str, Any]:
        for _ in range(400):
            length = rng.randint(lo, hi)
            actions = [rng.randint(0, 3) for _ in range(length)]
            if sum(1 for a in actions if machine[a] >= 2) < min_complex:
                continue
            init = (rng.randint(0, p - 1), rng.randint(0, p - 1))
            traj = cls.rollout(machine, init, actions, p)
            return {
                "initial": init,
                "actions": actions,
                "final": traj[-1],
                "trajectory": traj,
            }
        raise RuntimeError("failed to sample register candidate")

    @classmethod
    def _examples_uniquely_identify_machine(
        cls,
        machine: Tuple[int, int, int, int],
        examples: List[Dict[str, Any]],
        p: int,
    ) -> bool:
        """True iff only `machine` is consistent with all I/O pairs (candidates always fit `machine`)."""
        for m in cls.MACHINES:
            if m == machine:
                continue
            for ex in examples:
                if cls.final_state(m, ex["initial"], ex["actions"], p) != ex["final"]:
                    break
            else:
                return False
        return True

    @classmethod
    def _select_context(
        cls,
        rng: random.Random,
        machine: Tuple[int, int, int, int],
        p: int,
        ctx_lo: int,
        ctx_hi: int,
        min_complex: int,
        max_attempts: int = 500,
    ) -> List[Dict[str, Any]]:
        for _ in range(max_attempts):
            pool = [
                cls._make_candidate(rng, machine, p, ctx_lo, ctx_hi, min_complex)
                for _ in range(96)
            ]
            for _ in range(250):
                picks = rng.sample(range(len(pool)), k=cls.K_REGISTER_DEMOS)
                chosen = [pool[i] for i in picks]
                if cls._examples_uniquely_identify_machine(machine, chosen, p):
                    return chosen
        raise RuntimeError(
            "register_machine_2d: could not find identifying context examples"
        )

    def _fmt_register_trajectory(
        self, traj: List[Tuple[int, int]], actions: Sequence[int]
    ) -> str:
        """(x0, y0) -> A_j -> (x1, y1) -> ... -> final state."""
        parts: List[str] = [_fmt_state(traj[0])]
        for t, a in enumerate(actions):
            parts.append(self.ACTION_LABELS[int(a)])
            parts.append(_fmt_state(traj[t + 1]))
        return " -> ".join(parts)

    def generate(self, seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
        rng = random.Random(seed)
        cfg = self.DIFFICULTY[difficulty]
        p = cfg["p"]
        ctx_lo, ctx_hi = cfg["ctx_horizon"]
        min_c = cfg["min_complex_per_seq"]
        machine = rng.choice(self.MACHINES)
        demos = self._select_context(rng, machine, p, ctx_lo, ctx_hi, min_c)
        serial_demos = [
            {
                "initial": list(d["initial"]),
                "actions": list(d["actions"]),
                "final": list(d["final"]),
                "trajectory": [list(s) for s in d["trajectory"]],
            }
            for d in demos
        ]
        hidden_mapping = {
            self.ACTION_LABELS[i]: self.INSTRUCTION_NAMES[machine[i]] for i in range(4)
        }
        return {
            "difficulty": difficulty,
            "p": p,
            "hidden_mapping": hidden_mapping,
            "solved_examples": serial_demos,
            "_machine": machine,
            "_demos": demos,
        }

    def _reg_fmt_demonstration(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        demos = struct["_demos"]
        lines = []
        for demo in demos:
            r1, r2 = demo["initial"]
            r1f, r2f = demo["final"]
            act = " ".join(self.ACTION_LABELS[a] for a in demo["actions"])
            lines.append(
                f"Initial state: ({r1}, {r2}), action sequence: {act}, "
                f"final state: ({r1f}, {r2f})"
            )
        ctx = (
            "The following examples show a 2D dynamical system.\n"
            "At each step, one of the actions A0, A1, A2, or A3 is applied to update the state.\n"
            "Each example gives an initial state, the action sequence, and the resulting final state.\n\n"
            + "\n".join(lines)
            + "\n\n"
        )
        return ctx, [{"format": "demonstration", "demos": demos}]

    def _reg_fmt_trace(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        demos = struct["_demos"]
        blocks = []
        for demo in demos:
            traj_str = self._fmt_register_trajectory(
                demo["trajectory"], demo["actions"]
            )
            blocks.append(f"Trajectory: {traj_str}")
        ctx = (
            "The following trajectories show the step-by-step evolution of a 2D dynamical system.\n"
            "Each line alternates state and action: (x, y), then the action applied (A0–A3), "
            "then the next state, and so on, ending with a state.\n\n"
            + "\n".join(blocks)
            + "\n\n"
        )
        return ctx, [{"format": "trace", "demos": demos}]

    def _reg_fmt_natural_language(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        p = struct["p"]
        machine = struct["_machine"]
        parts = [
            f"This is a deterministic 2D dynamical system modulo {p}. "
            f"At any time the state is a pair with both components in "
            f"{{0, 1, ..., {p - 1}}}. Exactly one of the actions A0, A1, A2, A3 is applied per step. "
            "How each action updates the state:",
        ]
        for i in range(4):
            parts.append(f"{self.ACTION_LABELS[i]}: {self._STEP_VERBAL[machine[i]]}")
        ctx = "\n".join(parts) + "\n\n"
        return ctx, [{"format": "declarative_nl"}]

    def _reg_fmt_formal_spec(
        self, struct: Dict[str, Any], rng: random.Random
    ) -> Tuple[str, List[Any]]:
        p = struct["p"]
        machine = struct["_machine"]
        lines = [
            f"A deterministic 2D dynamical system over integers modulo {p}:",
            "  Each step applies exactly one of the transformations (actions) A0, A1, A2, or A3:",
        ]
        for i in range(4):
            inst = machine[i]
            lines.append(
                f"  {self.ACTION_LABELS[i]}: {self._STEP_FORMAL[inst]}  (mod {p})"
            )
        ctx = "\n".join(lines) + "\n\n"
        return ctx, [{"format": "formal_spec"}]

    def format(
        self, struct: Dict[str, Any], fmt: str, rng: random.Random
    ) -> Tuple[str, List[Any]]:
        dispatchers = {
            "demonstration": self._reg_fmt_demonstration,
            "natural_language": self._reg_fmt_natural_language,
            "trace": self._reg_fmt_trace,
            "formal_spec": self._reg_fmt_formal_spec,
        }
        return dispatchers[fmt](struct, rng)

    def _sample_query(
        self, struct: Dict[str, Any], rng: random.Random, ood: bool
    ) -> Tuple[Tuple[int, int], List[int]] | None:
        p = struct["p"]
        machine = struct["_machine"]
        demos = struct["_demos"]
        cfg = self.DIFFICULTY[struct["difficulty"]]
        lo, hi = cfg["ood_horizon"] if ood else cfg["iid_horizon"]
        min_c = cfg["min_complex_per_seq"]
        for _ in range(400):
            length = rng.randint(max(lo, 1), hi)
            actions = [rng.randint(0, 3) for _ in range(length)]
            if sum(1 for a in actions if machine[a] >= 2) < min_c:
                continue
            init = (rng.randint(0, p - 1), rng.randint(0, p - 1))
            if any(
                d["initial"] == init and list(d["actions"]) == list(actions)
                for d in demos
            ):
                continue
            return init, actions
        return None

    def _probe_impl(
        self,
        struct: Dict[str, Any],
        held_out: List[Any],
        rng: random.Random,
        *,
        ood: bool,
        kind: str,
    ) -> Tuple[str, str]:
        sampled = self._sample_query(struct, rng, ood)
        if sampled is None:
            raise RuntimeError(
                f"register_machine_2d probe exhausted ({kind}, ood={ood})"
            )
        init, actions = sampled
        p = struct["p"]
        machine = struct["_machine"]
        act_str = " ".join(self.ACTION_LABELS[a] for a in actions)
        if kind == "forward":
            fin = self.final_state(machine, init, actions, p)
            q = (
                f"Question: Starting from state {_fmt_state(init)}, "
                f"apply the action sequence {act_str} in order. "
                f"What is the final state?\n\n"
                f"{_PROBE_SINGLE_STATE_REPLY_SPEC}\n\n"
                f"Answer: "
            )
            return q, _fmt_state(fin)
        if kind == "traj":
            traj = self.rollout(machine, init, actions, p)
            q = (
                f"Question: Starting from state {_fmt_state(init)}, "
                f"apply the action sequence {act_str} in order. "
                f"List every state from step 0 through step {len(actions)} in order.\n\n"
                f"{_PROBE_TRAJECTORY_REPLY_SPEC}\n\n"
                f"Answer: "
            )
            return q, _fmt_trajectory(traj)
        if kind == "inverse":
            final_s = self.final_state(machine, init, actions, p)
            init_gold = self.inverse_initial_state(machine, final_s, actions, p)
            q = (
                f"Question: The action sequence {act_str} was applied in order. "
                f"After these steps, the state is {_fmt_state(final_s)}. "
                f"What was the initial state before the first step?\n\n"
                f"{_PROBE_SINGLE_STATE_REPLY_SPEC}\n\n"
                f"Answer: "
            )
            return q, _fmt_state(init_gold)
        raise ValueError(f"unknown probe kind {kind!r}")

    def probe(
        self,
        struct: Dict[str, Any],
        held_out: List[Any],
        application: str,
        rng: random.Random,
    ) -> Tuple[str, str]:
        table = {
            "forward": (False, "forward"),
            "inverse": (False, "inverse"),
            "ood@forward": (True, "forward"),
            "ood@traj": (True, "traj"),
            "ood@inverse": (True, "inverse"),
        }
        ood, kind = table[application]
        return self._probe_impl(struct, held_out, rng, ood=ood, kind=kind)


_register_task = RegisterMachine2DTask()

REGISTER_INSTRUCTION_NAMES = RegisterMachine2DTask.INSTRUCTION_NAMES
REGISTER_ACTION_LABELS = RegisterMachine2DTask.ACTION_LABELS
REGISTER_MACHINES = RegisterMachine2DTask.MACHINES
K_REGISTER_DEMOS = RegisterMachine2DTask.K_REGISTER_DEMOS
REGISTER_DIFFICULTY = RegisterMachine2DTask.DIFFICULTY


def register_step(
    machine: Sequence[int], state: Tuple[int, int], action: int, p: int
) -> Tuple[int, int]:
    return RegisterMachine2DTask.step(machine, state, action, p)


def register_rollout(
    machine: Sequence[int],
    initial_state: Tuple[int, int],
    action_sequence: Sequence[int],
    p: int,
) -> List[Tuple[int, int]]:
    return RegisterMachine2DTask.rollout(machine, initial_state, action_sequence, p)


def register_final_state(
    machine: Sequence[int],
    initial_state: Tuple[int, int],
    action_sequence: Sequence[int],
    p: int,
) -> Tuple[int, int]:
    return RegisterMachine2DTask.final_state(machine, initial_state, action_sequence, p)


def register_inverse_initial_state(
    machine: Sequence[int],
    final_state: Tuple[int, int],
    action_sequence: Sequence[int],
    p: int,
) -> Tuple[int, int]:
    return RegisterMachine2DTask.inverse_initial_state(
        machine, final_state, action_sequence, p
    )


def generate_register_machine_2d(seed: int = 42, difficulty: int = 1) -> Dict[str, Any]:
    return _register_task.generate(seed, difficulty)


def format_register_machine_2d(
    struct: Dict[str, Any], fmt: str, rng: random.Random
) -> Tuple[str, List[Any]]:
    return _register_task.format(struct, fmt, rng)


def probe_register_machine_2d(
    struct: Dict[str, Any],
    held_out: List[Any],
    application: str,
    rng: random.Random,
) -> Tuple[str, str]:
    return _register_task.probe(struct, held_out, application, rng)


__all__ = [
    "AFFINE2D_DIFFICULTY",
    "AFFINE2D_PROBE_APPLICATIONS",
    "AffineDynamics2DTask",
    "K_DEMOS",
    "K_REGISTER_DEMOS",
    "REGISTER_DIFFICULTY",
    "REGISTER_INSTRUCTION_NAMES",
    "REGISTER_MACHINES",
    "RegisterMachine2DTask",
    "format_affine_dynamics_2d",
    "format_register_machine_2d",
    "generate_affine_dynamics_2d",
    "generate_register_machine_2d",
    "probe_affine_dynamics_2d",
    "probe_register_machine_2d",
    "register_final_state",
    "register_inverse_initial_state",
    "register_rollout",
    "register_step",
]
