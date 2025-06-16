---
menu:
  tutorials:
    identifier: pytorch
    parent: integration-tutorials
title: PyTorch
weight: 1
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1opD2znuZRNSDbj3EwpwLwVVqJgCsg_Ei?usp=sharing" >}}

Use [Weights & Biases](https://wandb.com) for machine learning experiment tracking, dataset versioning, and project collaboration.

{{< img src="/images/tutorials/huggingface-why.png" alt="An image of the various capabilities that Weights & Biases provides, including experiment tracking, reports, and dataset and model versioning." >}}

## What this notebook covers

PyTorch is an open-source Python framework used to build and train deep learning models. Using W&B's Python library, you can log various metrics from your PyTorch model training runs to your W&B account. This allows you to review visual representations of your training data and track your model's performance more easily over subsequent runs.


{{< img src="/images/tutorials/pytorch.png" alt="" >}}

This tutorial guides you through how to import the W&B's Python library into a basic PyTorch training pipeline script that trains a model on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) (a database of handwritten digits). By the end of the tutorial, you will be able to review metrics from the model's training in your W&B account.

## Prerequisites

Before starting this tutorial, you need:

* [A W&B account](https://wandb.ai/site/).
* [A W&B API key]({{< relref "/support/find_api_key" >}}). This gives the script access to your W&B account so that it can log the training's metrics.
* Familiarity with and access to a notebook interface, such as [Jupyter](https://realpython.com/jupyter-notebook-introduction) or [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=GJBs_flRovLc). Each code example in this tutorial is a new code cell in a notebook.

## Step 1: Install the W&B library

To begin, install the [`wandb` Python library]({{< relref "/ref/python" >}}) in your environment.

```
!pip install wandb -Uq
```

The `U` flag instructs `pip` to upgrade the package to its latest version, and the `q` flag suppresses the installation output.

## Step 2: Install PyTorch and additional libraries for training pipeline

This example PyTorch training pipeline requires several additional libraries to perform the training, including:

* `os`: to interact with the operating system.
* `random`: to generate random numbers.
* [`torch`](https://docs.pytorch.org/docs/stable/index.html): the core PyTorch library for building and training models.
* `[onnx](https://onnx.ai/onnx/intro/concepts.html)`: to export the trained model to the [ONNX format].
* `[numpy](https://numpy.org/doc)`: for numerical operations.
* `[torchvision](https://docs.pytorch.org/vision/stable/index.html)`: to handle MNIST image dataset and transformations.
* `[tqdm](https://tqdm.github.io)`: a library for displaying progress bars in loops.

Install these libraries in your notebook environment:

```
!pip install torch torchvision onnx numpy tqdm -Uq
```

The command installs the required libraries. You do not need to specifically install `os` or `random` as they are part of Python's standard library.

## Step 3: Import libraries and configure the environment

In the next code cell, add the following code block to import the necessary libraries, configure the environment for training reproducibility, and set the device for training (either CPU or GPU):

```python
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# Sets the random seed for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Configures the device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Removes the slowest mirror from the list of MNIST dataset mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]
```

In addition to importing the libraries you installed in the previous step, this code block also:

* Sets the random seed for reproducibility, ensuring that the training results are consistent across runs.
* Configures the device for training, using your local machine's GPU if available or falling back to the CPU.
* Removes a slow mirror from the list of MNIST dataset mirrors to speed up data loading.

Running this code returns no output, indicating that the libraries were imported successfully and the environment was configured without errors.

## Step 4: Log in to W&B

Next, import the `wandb` library and log in to your W&B account using your API key:

```python
import wandb

wandb.login()
```

In this example, the `WANDB_API_KEY` environment variable is used to retrieve your API key. If you haven't set this variable, the function prompts you to enter your API key manually. We highly recommend using environment variables to store sensitive information like API keys to keep from exposing them in your code.

## Step 5: Define the configuration parameters and model pipeline

After logging in to W&B, configure your model's hyperparameters and metadata as a dictionary. You can adjust these hyperparameters over subsequent runs to experiment with different model configurations.

```python
config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN")
```

For the purposes of this tutorial, we've defined only a few hyperparameters in the `config` dictionary and have hardcoded the rest in the subsequent functions, but you can add more hyperparameters to the config as needed.

Next, define the overall model pipeline. The following function creates a project in your W&B account called `pytorch-demo` (if one doesn't already exist), sends the configuration parameters to W&B, and then runs a standard PyTorch training pipeline in the context of W&B. This allows you to track the model's training progress and performance metrics in the W&B dashboard.

```python
def model_pipeline(hyperparameters):

    # Initializes W&B and creates or updates the `pytorch-demo` project 
    with wandb.init(project="pytorch-demo", config=hyperparameters):

      # Retrieves the configuration parameters from W&B.
      config = wandb.config
    
      # Creates the model, data loaders, loss function, and optimizer objects
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # Trains the model and logs the results
      train(model, train_loader, criterion, optimizer, config)

      # Test the model and log the results  
      test(model, test_loader)

    return model
```

Specifically, the `model_pipeline()` function does the following:
1. Initializes W&B using `[wandb.init()]({{< relref "/ref/python/init" >}})`, and creates or updates the `pytorch-demo` project with the specified configuration parameters.
2. Retrieves the configuration parameters from the W&B run using `wandb.config()` to ensure that the model is trained with the correct hyperparameters.
3. Calls the `make` function to create the model, data loaders, loss function, and optimizer objects.
4. Trains the model using the `train` function, which tracks gradients and logs training metrics to W&B.
5. Tests the model's performance using the `test` function, which logs loss to W&B and saves the model in the ONNX format.

### Step 6: Assemble the data, model, loss function, and optimizer

The `make()` function returns all of the objects the model pipeline needs to operate. You can think of these objects as the ingredients you need before you start cooking. In this case, the ingredients are the data, model, loss function, and optimizer.

```python
def make(config):
    # Make the data loaders
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer
```

The `make()` function returns the objects by invoking several helper functions. The purpose of these helper functions and their roles in training the model are described in the following sections.

### Load the MNIST dataset and create data loaders

Load the MNIST dataset and create PyTorch data loaders for training and testing using the following `get_data()` and `make_loader()` functions.

```python
def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train,
                                              transform=transforms.ToTensor(),
                                              download=True)
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))

    return sub_dataset

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader
```

The `get_data()` function loads the MNIST dataset and creates a subset of the dataset by using every fifth image. This is done to reduce the dataset size for faster training during this tutorial, but results in the model having lower accuracy. You can increase the model's accuracy by using the full dataset.

The `make_loader()` function creates a PyTorch data loader from the subset of the MNIST dataset, which shuffles the data and creates batches of data for training.

### Define the model architecture

Define the model's architecture using the `ConvNet` class. The following code implements a standard [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) for image classification. It consists of two convolutional layers followed by a fully connected layer.

```python
class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
```

You can experiment with changing the model architecture by changing the number of convolutional layers, the number of filters in each convolutional layer, and the number of neurons in the fully connected layer.

### Define the training and testing functions

Next, specify the training logic in the `train()` function to track the gradients and parameters, and log your training metrics to W&B. 

The training function uses two important `wandb` methods: `[wandb.watch()]({{< relref "/ref/python/watch" >}})` and `[wandb.log()]({{< relref "/ref/python/log" >}})`. The `wandb.watch()` method tracks the model's gradients and weights, while the `wandb.log()` method logs the training metrics to W&B.

```python
def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more.
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss
```

The `train()` function iterates over the training data, while computing the loss, and updating the model's parameters. While training, the function invokes the `train_log()` function, which uses `wandb.log()` to log the loss to W&B and prints the loss after every 25 batches, along with the number of examples seen so far and the current epoch. While this example only logs the loss, you can log other data from your run, such as accuracy, scalars, and tables.

```python
def train_log(loss, example_ct, epoch):
    # Log loss metrics to W&B
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

Lastly, define the `test()` function, which computes the accuracy of the model on the test data, logs the accuracy to W&B, and saves the model in the ONNX format to your W&B project.

```python
def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")

        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

You can access the saved model file in from the[ **Files** section]({{< relref "/guides/track/project-page/#artifacts-tab" >}}) of your W&B project. This allows you to download the model and use it in other applications.

## Step 7: Run the model pipeline

Finally, run the model pipeline by calling the `model_pipeline()` function. This function builds the model, trains it, and logs the results to W&B.

```python
# Build, train and analyze the model with the pipeline
model = model_pipeline(config)
```

When you run the pipeline, the output provides links directly to your W&B project where you can view the model's training progress and performance metrics in real-time. The output links look like this:

```
View project at https://wandb.ai/*ACCOUNT-USER-NAME*/pytorch-demo
View run at https://wandb.ai/**ACCOUNT-USER-NAME**/pytorch-demo/runs/lvqcsfl4
```

## Use a Hyperparameter Sweep to Optimize the Model (Optional)

Hyperparameter sweeps allow you to define a range of hyperparameters and automatically run multiple training runs to find the best combination of hyperparameters for your model. This is particularly useful for optimizing model performance.

To do this, define a range of hyperparameters in a sweep configuration file, and then pass the file to the [`wandb.sweep()` method]({{< relref "/ref/python/sweep" >}}). 

````python
sweep_id = wandb.sweep(sweep_config)
```

The function returns a sweep ID, which you can use to start the sweep. Then pass the sweep ID to the [`wandb.agent()` method ]({{< relref "/ref/python/agent" >}}) to start the sweep.

```python
wandb.agent(sweep_id, function=train)
```

The results of the sweep are logged to your W&B project, allowing you to compare the performance of different hyperparameter combinations and select the best one.


## Advanced Setup Options

There are a few advanced setup options you can use to customize and improve your W&B experience.

* [Environment variables]({{< relref "/guides/hosting/env-vars/" >}}): Set API keys in environment variables so you can run training on a managed cluster.
* [Offline mode]({{< relref "/support/kb-articles/run_wandb_offline.md" >}}): Use `dryrun` mode to train offline and sync results later.
* [On-prem]({{< relref "/guides/hosting/hosting-options/self-managed" >}}): Install W&B in a private cloud or air-gapped servers in your own infrastructure.
* [Sweeps]({{< relref "/guides/models/sweeps/" >}}): Set up a hyperparameter search with our lightweight tool for tuning.
