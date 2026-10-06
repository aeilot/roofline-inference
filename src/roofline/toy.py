"""Toy-scale validation for the I/O bounds (design notes 6.8).

Element-level computation graphs and an exact red-blue pebble game solver,
plus the I/O cost of explicit schedules at real sizes. Lower bounds must
never exceed either: the solver checks the game and the compulsory term on
graphs small enough to search; the schedules check the pebble constants at
sizes where those terms bind.

Game (Hong & Kung 1981, with recomputation): at most S red pebbles; load
(blue -> red, cost 1), store (red -> blue, cost 1), compute (all parents red,
cost 0), delete red (cost 0). Start: blue on inputs. Goal: blue on outputs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Dag:
    parents: tuple[tuple[int, ...], ...]  # parents[v]; inputs have none
    outputs: tuple[int, ...]
    names: tuple[str, ...] = ()

    @property
    def inputs(self) -> tuple[int, ...]:
        return tuple(v for v, p in enumerate(self.parents) if not p)


class _DagBuilder:
    def __init__(self) -> None:
        self.parents: list[tuple[int, ...]] = []
        self.names: list[str] = []

    def add(self, name: str, *parents: int) -> int:
        self.parents.append(tuple(parents))
        self.names.append(name)
        return len(self.parents) - 1


def matmul_dag(m: int, n: int, k: int) -> Dag:
    """C = A @ B as FMA chains: c_ij^(p) = c_ij^(p-1) + a_ip * b_pj."""
    b = _DagBuilder()
    A = [[b.add(f"a{i}{p}") for p in range(k)] for i in range(m)]
    B = [[b.add(f"b{p}{j}") for j in range(n)] for p in range(k)]
    outs = []
    for i in range(m):
        for j in range(n):
            acc = b.add(f"c{i}{j}.0", A[i][0], B[0][j])
            for p in range(1, k):
                acc = b.add(f"c{i}{j}.{p}", acc, A[i][p], B[p][j])
            outs.append(acc)
    return Dag(tuple(b.parents), tuple(outs), tuple(b.names))


def attention_dag(n: int, L: int, d: int) -> Dag:
    """Unnormalized attention o_i = sum_j exp(q_i . k_j) v_j with head_dim d
    (value width 1): score FMA chains, then an FMA chain over keys per query.
    Exp and normalization are folded into the vertices; they do not change
    the level-1 structure the attention bound uses."""
    b = _DagBuilder()
    Q = [[b.add(f"q{i}{l}") for l in range(d)] for i in range(n)]
    K = [[b.add(f"k{j}{l}") for l in range(d)] for j in range(L)]
    V = [b.add(f"v{j}") for j in range(L)]
    outs = []
    for i in range(n):
        acc = None
        for j in range(L):
            s = b.add(f"s{i}{j}.0", Q[i][0], K[j][0])
            for l in range(1, d):
                s = b.add(f"s{i}{j}.{l}", s, Q[i][l], K[j][l])
            acc = b.add(f"o{i}.{j}", s, V[j]) if acc is None else b.add(f"o{i}.{j}", acc, s, V[j])
        outs.append(acc)
    return Dag(tuple(b.parents), tuple(outs), tuple(b.names))


def optimal_io(dag: Dag, S: int, limit: int = 5_000_000) -> int:
    """Minimum loads + stores of any complete calculation with S red pebbles
    (0-1 BFS over (red, blue) states). Raises if the search exceeds `limit`
    states or no calculation exists."""
    V = len(dag.parents)
    pmask = [sum(1 << p for p in ps) for ps in dag.parents]
    is_input = [not ps for ps in dag.parents]
    goal = sum(1 << v for v in dag.outputs)
    start = (0, sum(1 << v for v in dag.inputs))
    dist = {start: 0}
    q = deque([start])
    while q:
        state = q.popleft()
        red, blue = state
        cost = dist[state]
        if blue & goal == goal:
            return cost
        if len(dist) > limit:
            raise RuntimeError("state limit exceeded")
        full = red.bit_count() >= S
        for v in range(V):
            bit = 1 << v
            nxt = []
            if red & bit:
                nxt.append(((red & ~bit, blue), 0))  # delete
                if not blue & bit:
                    nxt.append(((red, blue | bit), 1))  # store
            elif not full:
                if blue & bit:
                    nxt.append(((red | bit, blue), 1))  # load
                if not is_input[v] and red & pmask[v] == pmask[v]:
                    nxt.append(((red | bit, blue), 0))  # compute
            for s2, c in nxt:
                d2 = cost + c
                if d2 < dist.get(s2, math.inf):
                    dist[s2] = d2
                    (q.appendleft if c == 0 else q.append)(s2)
    raise RuntimeError("no complete calculation (S too small?)")


# --- explicit schedules at real sizes (upper bounds on optimal I/O) -------------


def tiled_gemm_io(m: int, n: int, k: int, S: float, P: int = 1) -> float:
    """I/O (elements) of a square-tiled GEMM: each processor owns b x b output
    tiles (b^2 + 2b <= S), streams one column of A and one row of B per step,
    writes C once. Reads A n/b times and B m/b times."""
    b = max(1, int((math.sqrt(1 + S) - 1)))
    while b * b + 2 * b > S:
        b -= 1
    b = min(b, m, n)
    tiles_m, tiles_n = math.ceil(m / b), math.ceil(n / b)
    return tiles_n * m * k + tiles_m * k * n + m * n  # P cores split the tiles; totals add up


def flash_attention_io(n: int, L: int, d: int, S: float, causal: bool = False) -> tuple[float, int]:
    """I/O (elements) of FlashAttention-style tiling with value width d:
    query blocks of B rows stay on chip (Q block + output accumulator, 2Bd),
    K/V stream through in blocks of B_c rows (2 B_c d) with a B x B_c score
    tile. Every query block reads all keys it attends to. Returns (io, B)
    for the best B."""
    best = (math.inf, 0)
    Bc = 1
    largest = int((S - 2 * Bc * d) // (2 * d + Bc))  # largest B that fits
    for B in sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, n, largest}):
        if not 1 <= B <= n:
            continue
        if 2 * B * d + 2 * Bc * d + B * Bc > S:
            continue
        blocks = math.ceil(n / B)
        if causal:  # block i attends to keys < (i+1) B
            keys = sum(min(L, (i + 1) * B) for i in range(blocks))
        else:
            keys = blocks * L
        io = n * d + 2 * keys * d + n * d  # read Q, stream K and V, write O
        best = min(best, (io, B))
    return best
