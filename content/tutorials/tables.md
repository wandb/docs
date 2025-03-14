---
menu:
  tutorials:
    identifier: tables
    parent: null
title: Visualize predictions with tables
weight: 2
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W&B_Tables_Quickstart.ipynb" >}}

This guide explains how to track, visualize, and compare model predictions while using PyTorch with the Modified National Institute of Standards and Technology (MNIST) dataset.

Learn to:
- Log metrics, images, and text to a `wandb.Table()` during model training or evaluation.
- View, sort, filter, group, join, query, and explore these tables.
- Compare model predictions across specific images, hyperparameters, model versions, or time steps.

## Examples

### Compare predicted scores for specific images

[Live example: compare predictions after 1 vs. 5 epochs](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)

{{< img src="/images/tutorials/tables-1.png" alt="1 epoch vs. 5 epochs of training" >}}

The histograms compare per-class scores between two models. The top green bar shows the model "CNN-2, 1 epoch," trained for 1 epoch, and the bottom purple bar shows the model "CNN-2, 5 epochs." The images highlight disagreements. After 1 epoch, "4" scores high across all digits but scores highest on the correct label after 5 epochs.

### Focus on top errors over time

[Live example](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

View incorrect predictions by filtering rows where "guess" differs from "truth" on the full test data. There were 229 incorrect guesses after 1 epoch and 98 after 5 epochs.

{{< img src="/images/tutorials/tables-2.png" alt="side by side, 1 vs. 5 epochs of training" >}}

### Compare model performance and find patterns

[Full detail in a live example](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

Exclude correct answers and group by guess to view misclassified images and the distribution of true labels for two models side-by-side. A model variant with double the layer sizes and learning rate is on the left, and the baseline is on the right. The baseline makes slightly more mistakes for each guessed class.

{{< img src="/images/tutorials/tables-3.png" alt="grouped errors for baseline vs. double variant" >}}

## Sign up or log in

[Sign up or log in](https://wandb.ai/login) to W&B to interact with your experiments in the browser.

This example uses Google Colab as a hosted environment, but you can run your training scripts from anywhere and visualize metrics with W&B's experiment tracking tool.

```python
!pip install wandb -qqq
```

Log into your account

```python
import wandb
wandb.login()

WANDB_PROJECT = "mnist-viz"
```

## 0. Setup

Install dependencies, download the MNIST dataset, and create train and test datasets using PyTorch.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_dataloader(is_train, batch_size, slice=5):
    "Get a training dataloader"
    ds = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
    loader = torch.utils.data.DataLoader(dataset=ds, 
                                         batch_size=batch_size, 
                                         shuffle=True if is_train else False, 
                                         pin_memory=True, num_workers=2)
    return loader
```

## 1. Define the model and training schedule

- Set the number of epochs, each including a training and a validation step.
- Define a convolutional neural network following [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial).
- Load train and test sets using PyTorch.

```python
EPOCHS = 1
NUM_BATCHES_TO_LOG = 10
NUM_IMAGES_PER_BATCH = 32

NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L1_SIZE = 32
L2_SIZE = 64
CONV_KERNEL_SIZE = 5

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

For each epoch, complete a training and test step. Create a `wandb.Table()` at each test step to store predictions. You can visualize and query them in your browser.

```python
wandb.init(project="table-quickstart")

cfg = wandb.config
cfg.update({"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "l1_size": L1_SIZE, "l2_size": L2_SIZE,
            "conv_kernel": CONV_KERNEL_SIZE,
            "img_count": min(10000, NUM_IMAGES_PER_BATCH*NUM_BATCHES_TO_LOG)})

model = ConvNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break

total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))

    columns=["id", "image", "guess", "truth"] + ["score_" + str(digit) for digit in range(10)]
    test_table = wandb.Table(columns=columns)

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
        wandb.log({"epoch": epoch, "acc": acc})
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))

    wandb.log({"test_predictions": test_table})

wandb.finish()
```

## Next steps

Learn how to optimize hyperparameters using W&B Sweeps: [Optimize Hyperparameters]({{< relref "sweeps.md" >}})