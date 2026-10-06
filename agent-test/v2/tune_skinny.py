"""Tune skinny GEMM vs MLX matmul per (MT = ceil(M/8), N, K); writes skinny_table.json."""
import json, time, statistics
import mlx.core as mx
import kernels as K


def t(f, n=8):
    for _ in range(2):
        mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize(); s = time.perf_counter(); mx.eval(f()); mx.synchronize()
        ts.append(time.perf_counter() - s)
    return statistics.median(ts) * 1000


table = {}
for N, Kd in [(4096, 1024), (6144, 1024), (1024, 2048), (1024, 3072)]:
    Ws = [(mx.random.normal((N, Kd)) * 0.02).astype(mx.bfloat16) for _ in range(28)]
    mx.eval(Ws)
    for MT in range(2, 8):
        M = MT * 8
        x = mx.random.normal((M, Kd)).astype(mx.bfloat16)
        mx.eval(x)
        base = t(lambda: [x @ W.T for W in Ws])
        best = None
        for NT, NSG in [(1, 2), (1, 4), (1, 8), (2, 2), (2, 4)]:
            if N % (NT * 8 * NSG):
                continue
            K.SKINNY_CFG[(MT, N)] = (NT, NSG)
            ms = t(lambda: [K.skinny_matmul(x, W) for W in Ws])
            if best is None or ms < best[0]:
                best = (ms, NT, NSG)
        use = best[0] < 0.95 * base
        table[f"{MT},{N},{Kd}"] = dict(use=use, NT=best[1], NSG=best[2], skinny_ms=best[0], mlx_ms=base)
        print(N, Kd, M, f"mlx {base:.2f} skinny {best[0]:.2f}", "USE" if use else "", flush=True)
json.dump(table, open("skinny_table.json", "w"), indent=1)
