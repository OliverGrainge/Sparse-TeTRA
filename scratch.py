from model import ViT, SparseTernaryViT
from model.aggregation import BoQ, SALAD, MixVPR, CosPlace, ConvAP, GeM
import torch 

model = SparseTernaryViT()
agg = BoQ(**{})

# Create random input image
img = torch.randn(16, 3, 224, 224, requires_grad=True)

# Forward pass
out = model(img, sparsity=0.99999999999)
print("Output shape after backbone:", out.shape)

out = agg(out)
print("Output shape after aggregation:", out.shape)

# Compute dummy loss and check backprop
loss = out.sum()
loss.backward()

# Check if gradients exist
print("\nGradient check:")
print("Input gradient exists:", img.grad is not None)
print("Model has gradients:", any(p.grad is not None for p in model.parameters()))
print("Aggregator has gradients:", any(p.grad is not None for p in agg.parameters()))





