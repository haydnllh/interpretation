import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from interpretation.explainer.nn.nn_vis import NNVis

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
vis = NNVis(model)

fig, ax = plt.subplots(1, 4, figsize=(16, 4))

for i in range(1, 5):
    layer = f"layer{i}.0.conv2"
    
    img = vis.plot_visualization(
        layer_identifier=layer,
        input_shape=(3, 128, 128),
        channel_idx=0,
        max_iter=50,
        ax=ax[i-1]
    )
    
    ax[i-1].set_title(f"{layer}", fontsize=11, pad=8)

plt.tight_layout()
plt.show()