"""Quick paired comparison (not the formal benchmark): TTFT and decode vs mlx-lm."""
import os, sys, time, statistics, mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from engine import Engine
import thermal
model,_=load("Qwen/Qwen3-0.6B"); eng=Engine(model)
IDLE=float(os.environ.get("IDLE","0")); REP=int(os.environ.get("REP","4")); GEN=int(os.environ.get("GEN","65"))
def measure(it):
    st=[]; s=time.perf_counter(); toks=[]
    for t in it: st.append(time.perf_counter()-s); toks.append(t)
    mx.synchronize()
    g=[b-a for a,b in zip(st,st[1:])][2:]
    return st[0]*1000, statistics.median(g)*1000, toks
lengths=[int(a) for a in sys.argv[1:]] or [16,64,512,1024,4096]
for L in lengths:
    p=mx.random.randint(1000,150000,(L,),key=mx.random.key(10)).tolist()
    A=lambda: (t for t,_ in generate_step(mx.array(p),model,max_tokens=GEN))
    B=lambda: eng.generate(p,max_tokens=GEN)
    for f in (A,B): measure(f())
    res={"mlx":[], "v2":[]}
    thermal.wait_cool(idle=1)
    for r in range(REP):
        for name,f in ([("mlx",A),("v2",B)] if r%2==0 else [("v2",B),("mlx",A)]):
            mx.synchronize(); time.sleep(IDLE); res[name].append(measure(f()))
    same = res["mlx"][0][2]==res["v2"][0][2]
    pr=lambda k,i: statistics.median(x[i] for x in res[k])
    print(f"L={L} TTFT {pr('mlx',0):.1f} -> {pr('v2',0):.1f} ({pr('mlx',0)/pr('v2',0):.2f}x)  decode {pr('mlx',1):.2f} -> {pr('v2',1):.2f} ({pr('mlx',1)/pr('v2',1):.3f}x) tokens_equal={same} thermal={thermal.thermal_state()}", flush=True)
