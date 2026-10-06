"""Judge the 32k generated-sequence divergence with an FP32 forward (near-tie check)."""
import json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

d = json.load(open("results-long.json"))
row = next(r for r in d["runs"] if r["length"] == 32768)
a, b = row["measurements"]["mlx-lm"]["tokens"], row["measurements"]["v2"]["tokens"]
j = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
prompt = mx.random.randint(1000, 150000, (32768,), key=mx.random.key(10)).tolist()
model, _ = load("Qwen/Qwen3-0.6B")
model.set_dtype(mx.float32)
seq = mx.array(prompt + a[:j])[None]
c = make_prompt_cache(model)
for s in range(0, seq.shape[1] - 1, 2048):
    model(seq[:, s:min(s + 2048, seq.shape[1] - 1)], cache=c)
    mx.eval([x.state for x in c])
r = model(seq[:, -1:], cache=c)[0, -1]
gap = abs(r[a[j]] - r[b[j]]).item()
top = mx.sort(r)[-3:].tolist()
out = dict(position=j, mlx_token=a[j], v2_token=b[j], fp32_logit_mlx=r[a[j]].item(),
           fp32_logit_v2=r[b[j]].item(), fp32_gap=gap, fp32_top3=top,
           near_tie_threshold_from_check=0.2502, near_tie=gap < 0.2502)
print(json.dumps(out))
json.dump(out, open("div32k.json", "w"), indent=1)
