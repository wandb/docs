---
menu:
  tutorials:
    identifier: tables
    parent: null
title: Visualize predictions with tables
weight: 2
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

This guide explains how to track, visualize, and compare model predictions during training using PyTorch on MNIST data.

Learn how to:
1. Log metrics, images, and text to a `wandb.Table()` during model training or evaluation.
2. View, sort, filter, group, join, interactively query, and explore these tables.
3. Compare model predictions dynamically across images, hyperparameters, model versions, or time steps.

## Examples
### Compare predicted scores for specific images

[Live example: compare predictions after 1 vs. 5 epochs of training →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="1 epoch vs. 5 epochs of training" >}}

The histograms illustrate per-class scores between two models. The top green bar represents model "CNN-2, 1 epoch" (id 0), trained for one epoch. The bottom purple bar represents model "CNN-2, 5 epochs" (id 1), trained for five epochs. The filtered images display cases where the models disagree. For instance, in the first row, the "4" achieves high scores across all possible digits after one epoch, but after five epochs, it scores highest on the correct label and very low on others.

### Focus on top errors over time
[Live example →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

View incorrect predictions by filtering rows where "guess" != "truth" across the full test data. There are 229 wrong guesses after one training epoch and only 98 after five epochs.

{{< img src="/images/tutorials/tables-2.png" alt="side by side, 1 vs. 5 epochs of training" >}}

### Compare model performance and find patterns

[See full detail in a live example →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

Filter out correct answers, then group by the guess to view examples of misclassified images and the underlying distribution of true labels for two models side by side. A model variant with double the layer sizes and learning rate is on the left, while the baseline is on the right. The baseline makes slightly more mistakes for each guessed class.

{{< img src="/images/tutorials/tables-3.png" alt="grouped errors for baseline vs. double variant" >}}

## Sign up or log in

[Sign up or log in](https://wandb.ai/login) to W&B to see and interact with experiments in the browser.

This example uses Google Colab for convenience, but you can run training scripts from any location and visualize metrics with W&B's experiment tracking tool.

```python
!pip install wandb -qqq
```

Log into your account:

```python
import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

## 0. Setup

Install dependencies, download MNIST, and create train and test datasets using PyTorch.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create train and test dataloaders
def get_dataloader(is_train, batch_size, slice=5):
    "Retrieve a training dataloader."
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=is_train, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. Define the model and training schedule

* Set the number of epochs, where each epoch consists of a training step and a validation step. Optionally configure how much data to log per test step. For this example, the batch size and number of images per batch are set low to simplify the demo.
* Define a convolutional neural network following the [PyTorch tutorial](https://github.com/yunjey/pytorch-tutorial) code.
* Load train and test sets using PyTorch.

```python
# Number of epochs to run
EPOCHS = 1

# Number of batches to log from the test data for each test step
NUM_BATCHES_TO_LOG = 10 

# Number of images to log per test batch
NUM_IMAGES_PER_BATCH = 32 

# Training configuration and hyperparameters
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
CONV_KERNEL_SIZE = 5

# Define a two-layer convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, L1_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm2d(L1_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(L1_SIZE, L2_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm2d(L2_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*L2_SIZE, NUM_CLASSES)
        self.softmax = nn.Softmax(NUM_CLASSES)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

train_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
test_loader = get_dataloader(is_train=False, batch_size=2*BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 2. Run training and log test predictions

For every epoch, run a training step and a test step. Create a `wandb.Table()` for each test step to store predictions. These can be visualized and compared side by side in the browser.

```python
# ✨ W&B: Initialize a new run to track this model's training
wandb.init(project="table-quickstart")

# ✨ W&B: Log hyperparameters using config
cfg = wandb.config
cfg.update({"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "l1_size": L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel": CONV_KERNEL_SIZE,
            "img_count": min(10000, NUM_IMAGES_PER_BATCH * NUM_BATCHES_TO_LOG)})

# Define model, loss, and optimizer
model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Convenience function to log predictions for a batch of test images
def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
    # Obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()

    _id = 0
    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        # Add required info to data table: id, image pixels, model's guess, true label, scores for all classes
        img_id = f"{_id}_{log_counter}"
        test_table.add_data(img_id, wandb.Image(i), p, l, *s)
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break

# Train the model
total_step = len(train_loader)
for epoch in range(EPOCHS):
    # Training step
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        # ✨ W&B: Log loss over training steps, visualized live in the UI
        wandb.log({"loss": loss})
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
            
    # ✨ W&B: Create a Table to store predictions for each test step
    columns = ["id", "image", "guess", "truth"]
    columns += [f"score_{digit}" for digit in range(10)]
    test_table = wandb.Table(columns=columns)

    # Test the model
    model.eval()
    log_counter = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if log_counter < NUM_BATCHES_TO_LOG:
                log_test_predictions(images, labels, outputs, predicted, test_table, log_counter)
                log_counter += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        # ✨ W&B: Log accuracy across training epochs to visualize in the UI
        wandb.log({"epoch": epoch, "acc": acc})
        print(f'Test accuracy of the model on the 10,000 test images: {acc:.2f} %')

    # ✨ W&B: Log predictions table to wandb
    wandb.log({"test_predictions": test_table})

# ✨ W&B: Mark the run as complete (useful for multi-cell notebook)
wandb.finish()
```

## What's next?
In the next tutorial, learn how to optimize hyperparameters using W&B Sweeps:
## 👉 [Optimize Hyperparameters]({{< relref "sweeps.md" >}})