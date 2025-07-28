import os, sys, time, torch

LVL = int(os.getenv("DEBUG", "0") or 0)

def dprint(level: int, *msg):
    if LVL >= level:
        print(f"[D{level}]", *msg, file=sys.stderr)
        
def tstats(t, name="tensor", level=2, n=5):
    if LVL < level: return
    with torch.no_grad():
        flat = t.detach().float().view(-1)
        preview = flat[:n].cpu().tolist()
        dprint(level, f"{name}: shape={tuple(t.shape)}, mean={flat.mean():.3g}, std={flat.std():.3g}, "
                 f"min={flat.min():.3g}, max={flat.max():.3g}, sample={preview}")