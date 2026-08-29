import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from interpretation.explainer.nn.saliency import SaliencyMap

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model.eval()

saliency = SaliencyMap(model)
layer = "layer4.0.conv2"

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform
)

sample = 6
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

img = saliency.plot_map(
    trainset[sample][0].unsqueeze(0).numpy(),
    int(trainset[sample][1]),
    ax=ax[1]
) 

ax[0].imshow(trainset[sample][0].permute(1,2,0).numpy())
ax[0].axis('off')
plt.show() 