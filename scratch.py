
from model.baselines import DinoBoQ, DinoSalad, CosPlace, EigenPlaces, MixVPR
import torch
import torch.profiler

model = DinoBoQ()
input = torch.randn(1, 3, 322, 322)
with torch.no_grad():
    model(input)  # warm-up

with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True) as prof:
    with torch.no_grad():
        model(input)

events = prof.key_averages()


SPARSITY = 0.40
adjusted_flops = 0
for evt in events:
    if "linear" in evt.key.lower() or "addmm" in evt.key.lower():   
        adjusted_flops += evt.flops * (1 - SPARSITY)
    else:
        print(evt.key.lower(), evt.flops)
        adjusted_flops += evt.flops
