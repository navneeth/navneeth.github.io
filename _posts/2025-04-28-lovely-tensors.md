---
layout: post
title: Debugging a Broken ImageNet Classifier with Lovely Tensors: A Visual Journey
---

## Introduction

Debugging deep learning models can often feel like navigating a dense forest of tensors. Understanding the shape, values, and gradients flowing through your network is crucial, but the default PyTorch printing can be overwhelming. Enter (`lovely-tensors`)[https://github.com/xl0/lovely-tensors], a delightful little library that provides concise and informative summaries of your tensors, making debugging a much more pleasant and visual experience.

In this tutorial, we'll intentionally build a poorly performing ImageNet classifier. Then, step by step, we'll leverage the power of `lovely-tensors` to pinpoint the issues and understand what's going wrong under the hood, with a focus on interpreting its insightful output. Get ready to transform your debugging workflow!

## Setting Up Our (Deliberately) Bad Classifier

First, let's set up our environment and define a simple convolutional neural network that's destined for failure on ImageNet. We'll use a standard ResNet-18 architecture but introduce some common pitfalls.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
import lovely_tensors as lt

lt.monkey_patch() # Enable lovely-tensors' magic!

# Let's create a ResNet-18 model
model = models.resnet18(num_classes=1000)

# Intentionally using a very small learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = torch.nn.CrossEntropyLoss()

# Using FakeData for simplicity (ImageNet is large!)
img_size = 224
train_dataset = FakeData(
    size=100,
    image_size=(3, img_size, img_size),
    num_classes=1000
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Let's run a few training steps
num_epochs = 3
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training finished. Let's try to debug why it's probably not great.")---
layout: post
title: Debugging a Broken ImageNet Classifier with Lovely Tensors: A Visual Journey
---

## Introduction

Debugging deep learning models can often feel like navigating a dense forest of tensors. Understanding the shape, values, and gradients flowing through your network is crucial, but the default PyTorch printing can be overwhelming. Enter `lovely-tensors`, a delightful little library that provides concise and informative summaries of your tensors, making debugging a much more pleasant and visual experience.

In this tutorial, we'll intentionally build a poorly performing ImageNet classifier. Then, step by step, we'll leverage the power of `lovely-tensors` to pinpoint the issues and understand what's going wrong under the hood, with a focus on interpreting its insightful output. Get ready to transform your debugging workflow!

## Setting Up Our (Deliberately) Bad Classifier

First, let's set up our environment and define a simple convolutional neural network that's destined for failure on ImageNet. We'll use a standard ResNet-18 architecture but introduce some common pitfalls.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
import lovely_tensors as lt

lt.monkey_patch() # Enable lovely-tensors' magic!

# Let's create a ResNet-18 model
model = models.resnet18(num_classes=1000)

# Intentionally using a very small learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = torch.nn.CrossEntropyLoss()

# Using FakeData for simplicity (ImageNet is large!)
img_size = 224
train_dataset = FakeData(
    size=100,
    image_size=(3, img_size, img_size),
    num_classes=1000
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Let's run a few training steps
num_epochs = 3
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training finished. Let's try to debug why it's probably not great.")
