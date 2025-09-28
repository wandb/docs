---
menu:
  tutorials:
    identifier: pytorch
    parent: integration-tutorials
title: PyTorch
weight: 1
---
# Integrate PyTorch with Weights & Biases

Use [Weights & Biases (W&B)](https://wandb.ai) to track machine learning experiments, version datasets, and collaborate on projects.

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

{{< img src="/images/tutorials/huggingface-why.png" alt="Benefits of using W&B" >}}

## Overview

This tutorial shows you how to integrate W&B with PyTorch. After you complete it, you’ll be able to:

- Log hyperparameters and metadata  
- Track model gradients, parameters, and metrics  
- Save models and artifacts  
- Automate hyperparameter optimization with W&B Sweeps  

{{< img src="/images/tutorials/pytorch.png" alt="PyTorch and W&B integration diagram" >}}

## Before you begin

You need the following:

- Python 3.7 or later  
- PyTorch installed  
- A free [W&B account](https://wandb.ai)  
- GPU hardware (optional but recommended)  

## Quickstart

The following example shows how to add W&B tracking to a training loop:

```python
import wandb

# Start a new experiment
with wandb.init(project="new-sota-model") as run:
    # Log hyperparameters
    run.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    # Define model and data
    model, dataloader = get_model(), get_data()

    # Track gradients (optional)
    run.watch(model)

    for batch in dataloader:
        metrics = model.training_step()
        # Log metrics to visualize performance
        run.log(metrics)

    # Save trained model
    model.to_onnx()
    run.save("model.onnx")
```

For a walkthrough, see the [video tutorial](https://wandb.me/pytorch-video).

> **Note:** Steps labeled *Step X* show only the minimal code needed for W&B integration. Other sections cover model and data setup.

---

## Step 1. Install W&B

Install the libraries:

```python
!pip install wandb onnx -Uq
```

## Step 2. Import and log in

Import W&B and log in:

```python
import wandb

wandb.login()
```

If this is your first time, create an account at the link provided.

---

## Step 3. Define the experiment

Track hyperparameters and metadata with `wandb.init()`. Use a config dictionary for reproducibility:

```python
config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN"
)
```

A typical ML pipeline includes the following steps:

1. Build the model, data, and optimizer  
2. Train the model  
3. Test the performance  

```python
def model_pipeline(hyperparameters):
    with wandb.init(project="pytorch-demo", config=hyperparameters) as run:
        config = run.config

        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        train(model, train_loader, criterion, optimizer, config)
        test(model, test_loader)

    return model
```

---

## Step 4. Load data and define the model

Load the data:

```python
def get_data(slice=5, train=True):
    dataset = torchvision.datasets.MNIST(
        root=".",
        train=train,
        transform=transforms.ToTensor(),
        download=True
    )
    return torch.utils.data.Subset(dataset, range(0, len(dataset), slice))

def make_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
```

Define the model:

```python
class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
```

---

## Step 5. Train the model

Log gradients with `run.watch()` and metrics with `run.log()`:

```python
def train(model, loader, criterion, optimizer, config):
    run = wandb.init(project="pytorch-demo", config=config)
    run.watch(model, criterion, log="all", log_freq=10)

    total_batches = len(loader) * config.epochs
    example_ct, batch_ct = 0, 0

    for epoch in tqdm(range(config.epochs)):
        for images, labels in loader:
            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1

            if (batch_ct + 1) % 25 == 0:
                train_log(loss, example_ct, epoch)

def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```

Log metrics:

```python
def train_log(loss, example_ct, epoch):
    with wandb.init(project="pytorch-demo") as run:
        run.log({"epoch": epoch, "loss": loss}, step=example_ct)
        print(f"Loss after {example_ct} examples: {loss:.3f}")
```

---

## Step 6. Test and save the model

Evaluate and save the trained model:

```python
def test(model, test_loader):
    model.eval()
    with wandb.init(project="pytorch-demo") as run:
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f"Accuracy: {accuracy:.2%}")
            run.log({"test_accuracy": accuracy})

        torch.onnx.export(model, images, "model.onnx")
        run.save("model.onnx")
```

---

## Step 7. Run and monitor the experiment

Run the pipeline:

```python
model = model_pipeline(config)
```

Monitor the run in W&B:

- **Charts**: Gradients, parameters, and loss curves  
- **System**: CPU, GPU, and memory utilization  
- **Logs**: Console outputs  
- **Files**: Artifacts like `model.onnx`  

---

## Step 8. Optimize hyperparameters with Sweeps

Use W&B Sweeps to explore hyperparameters:

1. Define the sweep configuration:  
   ```python
   sweep_id = wandb.sweep(sweep_config)
   ```
2. Run the sweep agent:  
   ```python
   wandb.agent(sweep_id, function=train)
   ```

For a complete example, see the [Colab sweep notebook](https://wandb.me/sweeps-colab).

{{< img src="/images/tutorials/pytorch-2.png" alt="PyTorch training dashboard" >}}

---

## Example gallery

See tracked projects in the [W&B Gallery](https://app.wandb.ai/gallery).

## Advanced configuration

For advanced use cases, see:

- [Environment variables]({{< relref "/guides/hosting/env-vars/" >}})  
- [Offline mode]({{< relref "/support/kb-articles/run_wandb_offline.md" >}})  
- [On-premises hosting]({{< relref "/guides/hosting/hosting-options/self-managed" >}})  
- [Sweeps]({{< relref "/guides/models/sweeps/" >}})  
