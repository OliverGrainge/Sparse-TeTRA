from model import ViT 
import torch 

model = ViT(
    attention_linear_layer="SparseTernaryLinear",
    attention_linear_kwargs={
        "sparsity": 0.7
    },
    feedforward_linear_layer="SparseTernaryLinear",
    feedforward_linear_kwargs={
        "sparsity": 0.9
    }
)

print(model)
img = torch.randn(1, 3, 224, 224)

out = model(img)
print(out.shape)