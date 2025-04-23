from model.models import BoQModel
import torch
model = BoQModel()


img = torch.randn(1, 3, 224, 224)

print(model(img).shape)
