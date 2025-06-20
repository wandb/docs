---
menu:
  tutorials:
    identifier: pytorch
    parent: integration-tutorials
title: Add Weights & Biases experiment tracking to your PyTorch projects
weight: 1
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

Use [Weights & Biases](https://wandb.com) with PyTorch for machine learning experiment tracking, dataset versioning, and project collaboration.

{{< img src="/images/tutorials/huggingface-why.png" alt="Feature icons for W&B tools: Experiments, Reports, Artifacts, Tables, Sweeps, Launch, and Models." >}}

## What this notebook covers
This notebook shows you how to integrate Weights & Biases (W&B) with your PyTorch code to add experiment tracking to your machine learning pipeline.

{{< img src="/images/tutorials/pytorch.png" alt="Weights & Biases dashboard showing gradient visualization graphs that track model training progress." >}}

## Background
[PyTorch](https://pytorch.org/) is an open-source machine learning framework built for Python that provides high performance and scalability for production deployment, making it  popular for research and rapid prototyping. When you use W&B with your PyTorch projects, you can track your experiments, visualize results, and collaborate with your team in real time. Log metrics, hyperparameters, and model checkpoints to W&B as you train. Then view your results on the W&B Dashboard at [wandb.ai](http://wandb.ai).

W&B helps you track everything for your PyTorch projects - model architectures, datasets, and results. With W&B, you can share discoveries with your team and stay organized when you test different models and fine tune hyperparameters.

## TL;DR

In the following sections, you'll [set up your PyTorch environment](#set-up-your-environment) for W&B integration and [log in to the W&B library](#install-wb-and-log-in). Then, you'll define your experiment, [set up your data loading and model architecture](#configure-dataloaders-and-define-your-architecture), and [integrate W&B into the pipeline](#integrate-wb-into-your-pipeline) to track gradients and log metrics.

In a nutshell, the end-to-end process looks like this:

```python
# Import the library
import wandb

# Start a new experiment
wandb.init(project="pytorch-demo")

# Capture a dictionary of hyperparameters
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

# Set up model and data
model = get_model()
dataloader = get_data()

# Track gradients and parameters during training
wandb.watch(model, log="all", log_freq=10)

# Training loop
for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        # Forward pass, loss calculation, backward pass, etc.
        loss = train_batch(batch)
        accuracy = calculate_accuracy(batch)
        
        # Log metrics to visualize model performance in real-time
        wandb.log({
            "loss": loss.item(),
            "accuracy": accuracy,
            "epoch": epoch
        })

# Save the model in ONNX format
demo_input = torch.randn(1, 1, 28, 28, device=device)
torch.onnx.export(model, demo_input, "model.onnx")
wandb.save("model.onnx")
```

All metrics, gradients, and parameters that `wandb.log()` and `wandb.watch()` log appear in real-time on your [W&B Dashboard](http://wandb.ai).

# Detailed integration steps
Read on for comprehensive steps to integrate W&B with your PyTorch code. Follow along with the [video tutorial](http://wandb.me/pytorch-video) too.

## Set up your environment
Before you begin, you should already have [PyTorch and TorchVision](https://pytorch.org/get-started/) installed in your environment. The following script sets up your Python environment and PyTorch model with the configurations that you need to install W&B. If you've already configured your training pipeline, you can skip ahead to the [Installation](#install-wb-and-log-in) section. 

Copy the following setup script into a new file `wandb-setup.py`:

```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Remove slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [
    mirror
    for mirror in torchvision.datasets.MNIST.mirrors
    if not mirror.startswith("http://yann.lecun.com")
]
```

Execute the `wandb-setup.py` file in your environment:

```bash
python path/to/wandb-setup.py
```

The script completes the setup tasks silently. Your environment is now ready for you to install the Weights & Biases tracking code. 

## Install W&B and log in
Use `pip` to install the W&B library. We recommend that you also install ONNX if you haven't already. ONNX lets you export your model to the [Open Neural Network eXchange](https://onnx.ai/) standard format so you can deploy it across different platforms and frameworks. 

Use the following command to quietly (`-q`) install or upgrade (`-U`) both packages:

```bash
pip install wandb onnx -Uq
```

Next, authenticate with Weights & Biases so that you can log data to the W&B service. 

1. Import the `wandb` package:
   
```python
import wandb
```

2. Then run the login command:

```python
wandb.login()
```

3. When prompted, choose to log in with an existing account or create a new account for free. Follow the instructions to log in or complete your registration and obtain your API key.  W&B saves your credentials locally so you won't need to log in again on the same machine.

If you have trouble with the authentication step, see the support article [How can I resolve login issues with my account?](https://docs.wandb.ai/support/resolve_login_issues_with_account/)

## Define your experiment and pipeline
As a best practice, structure your machine learning experiments into clear, trackable pipelines. The experiment pipeline that we recommend comprises two phases:

1. [Set up hyperparameters and metadata](#set-up-hyperparameters-and-metadata)  
2. [Implement the model pipeline](#implement-the-model-pipeline)

This structured approach helps you maintain reproducible experiments, compare different versions and iterations of your models, and efficiently scale and debug your training workflows. You can also standardize the experimentation process across your team so it's easier to understand what works and what doesn't in your model development process.

### Set up hyperparameters and metadata
Hyperparameters and metadata help you organize and filter your experiments later, especially when you work with different architectures or datasets within the same project. Define the hyperparameters and metadata that describe your model. You can store these settings in a configuration dictionary that you'll access throughout your training process. 

Here's a basic configuration that you can pass into your project:

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

While this example shows just a few parameters, you can include any aspect of your model in the configuration. This flexibility allows you to track and compare different model variations as your experiments evolve.

### Implement the model pipeline
Once you've defined the metadata, create a function to initialize W&B tracking, set up your model components, and execute the training and testing phases. Use the `wandb.init()` context to establish communication between your code and the W&B servers. The function passes your configuration dictionary to `wandb.init()` and W&B logs your hyperparameter values. 

```python
def model_pipeline(hyperparameters):
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        config = wandb.config
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)
        train(model, train_loader, criterion, optimizer, config)
        test(model, test_loader)
    return model
```

When you start the experiment, W&B creates and shares the link to your new project so you can watch the magic right away.
 
```
wandb: ⭐️ View project at https://wandb.ai/joemama/pytorch-demo
wandb: 🚀 View run at https://wandb.ai/joemama/pytorch-demo/runs/abc123yz
```

For the most accurate logging, get your parameters from `wandb.config`. Your model pipeline already uses `wandb.config` as `config`, so you're ready to implement this function to create your components.

```python
def make(config):
    # Prepare your data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make your model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer
```

The `make()` function handles several key setup tasks. It prepares the training and test datasets with `get_data()` and `make_loader()`. It also initializes the PyTorch ConvNet model and moves it to the appropriate device, and sets up the loss function (Cross-Entropy Loss) and optimizer (Adam) for model training.

**Note:** For increased resilience, W&B runs your tracking code in separate processes. If you ever lose connection, use the `wandb sync` command once you're back online to upload any missing data.

## Configure DataLoaders and define your architecture
While W&B helps you track your experiments, your core PyTorch components remain unchanged. The following sections show an example of how you might set up your data loading and model architecture. If you've already prepared your data and architecture for PyTorch model training, you can skip ahead to [Integrate W&B into your pipeline](#integrate-wb-into-your-pipeline).

### Prepare your data for PyTorch model training
Before you start training, use the `get_data()` and `make_loader()` PyTorch functions to load your datasets and set up your PyTorch DataLoaders.

The `get_data()` function prepares the training and test datasets.

```python
def get_data(slice=5, train=True):
    # Load and preprocess MNIST dataset
    full_dataset = torchvision.datasets.MNIST(
        root=".", 
        train=train, 
        transform=transforms.ToTensor(), 
        download=True
    )
    # Create subset of data with specified slice
    sub_dataset = torch.utils.data.Subset(
        full_dataset, 
        indices=range(0, len(full_dataset), 
        slice)
    )
    return sub_dataset
```

The `make_loader()` function sets up the PyTorch DataLoaders.

```python
def make_loader(dataset, batch_size):
    # Configure DataLoader with specified batch size
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return loader
```

### Define the model architecture
Next, define your model architecture. This example uses a standard ConvNet, but you can modify the architecture to experiment with different approaches. 

```python
# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()

        # First convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Second convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected output layer
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
```

Experiment with different architectures, layer configurations, and activation functions. You can compare the performance of each variation at [wandb.ai](https://wandb.ai) once you integrate W&B into your pipeline.

## Integrate W&B into your pipeline
To integrate W&B into your training pipeline, you'll use two key functions: 
* **[wandb.watch()](#track-gradients-and-parameters-with-wandbwatch)** to track model gradients and parameters  
* **[wandb.log()](#log-metrics-with-wandblog)** to track everything else

### Track gradients and parameters with `wandb.watch()`
To keep track of the gradients and parameters for your model, call `wandb.watch()` before you start training. The `log_freq` parameter controls how often W&B logs these metrics. Once you've set up tracking, your training loop runs just like normal - iterating through epochs and batches, computing forward and backward passes, and updating weights with your optimizer.

```python
def train(model, loader, criterion, optimizer, config):
    # Track model gradients and parameters
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Initialize tracking variables
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples processed
    batch_ct = 0

    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):
            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1

            # Log metrics every 25 batches
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

    # Update weights
    optimizer.step()

    return loss
```

### Log metrics with `wandb.log()`
To log your training metrics, use `wandb.log`. The main difference between this option and and standard training loops is that, when you use `wandb.log`, you log metrics to W&B for visualization and analysis instead of printing them to your terminal. 

With `wandb.log`:
* The **keys** are strings that identify what you want to log  
* The **values** are the metrics that you want to track  
* The optional **`step` parameter** logs your training progression

The logging call in the following example creates visualizations in the W&B Dashboard that show the relationship between training progress and the epoch number and loss values.  

```python
def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
```

With `example_ct` (example count) as the step value, the call uses the number of processed examples as the step count. This is useful to compare across different batch sizes. You can also use batch count or epochs. For longer training runs, it might be more appropriate to log by epoch.

## Save and test your model
Once you've trained your model, test it to find out how well it performs on data it hasn't seen before. Run your model against a fresh validation dataset or test it with examples that match your production scenarios.

Then save your model's architecture and final parameters. To ensure compatibility across different platforms, export your model in the ONNX format. When you store your model on the W&B servers with `wandb.save()`, you can track which model files correspond to specific training runs. 

Here's how to test and save your model:

```python
def test(model, test_loader):
    model.eval()

    # Test your model's performance
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the model on the {total} "
            + f"test images: {correct / total:%}"
        )

        wandb.log({"test_accuracy": correct / total})

    # Save your model in ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
```

To download the saved ONNX model, navigate to the **Files** tab on your [W&B Dashboard](https://wandb.ai/). For advanced model management options for your machine learning assets (including model storage, versioning, and distribution), explore the [W&B Artifacts](https://www.wandb.com/artifacts) tools.

## Run your experiment and analyze the results
Now that you've built your PyTorch model pipeline and added W&B tracking, call `model_pipeline(config)` to run your experiment with full instrumentation.

```python
# Build, train, and analyze the model with the pipeline
model = model_pipeline(config)
```

When you make the call:

* W&B starts a new experiment and logs hyperparameters and metadata through `wandb.init()`.
* Your model begins to train, and `wandb.log()` captures metrics like loss and accuracy in real time.  
* Throughout training, `wandb.watch()` tracks gradients and parameters (see the live data on the [W&B Dashboard](https://wandb.ai/)). 
* When training completes, your model exports to ONNX format and saves to W&B.  
* W&B then displays a summary of your experiment results.

Navigate to [wandb.ai](http://wandb.ai) to analyze the results for your model configuration on the W&B Dashboard. For more information on navigating your project in the W&B Dashboard, see the [Projects](https://docs.wandb.ai/guides/track/project-page/) documentation.

## Next steps
Now that you've integrated Weights & Biases into your PyTorch model training pipeline, explore these additional W&B resources to further optimize your machine learning projects.

### Optimize your model with Sweeps
Now that you've run a single experiment, you'll want to try different hyperparameter combinations. [W&B Sweeps](https://docs.wandb.ai/guides/sweeps/) is a lightweight tuning tool that automates the exploration of different hyperparameter configurations. Sweeps can systematically explore various model configurations and track the results in your W&B Dashboard. This makes it easy to find the optimal hyperparameters for your PyTorch model. To learn more about how to run a hyperparameter sweep with Weights & Biases, see the [Organizing Hyperparameter Sweeps in PyTorch with W&B](http://wandb.me/sweeps-colab) Colab notebook.

### Explore advanced W&B configurations
Weights & Biases offers advanced setup options to fit specific needs:
* **[Environment variables](https://docs.wandb.ai/guides/hosting/env-vars/):** Set your W&B API keys in environment variables to run training on a managed cluster or CI/CD pipeline.  
* **[Offline mode](https://docs.wandb.ai/support/run_wandb_offline/):** Use the "dryrun" mode to train your models offline, then sync the results to W&B later when you're connected.  
* **[On-prem installations](https://docs.wandb.ai/guides/hosting/hosting-options/self-managed/):** For enterprise users or those with strict data governance requirements, install W&B in your own private cloud or air-gapped infrastructure.

Explore these and other advanced features in the [W&B platform documentation](https://docs.wandb.ai/guides/hosting).

### Browse example reports and related content
For examples of how teams use W&B to track and visualize their machine learning projects, search the [Fully Connected blog](https://www.wandb.com/blog) for posts that showcase different W&B visualizations and use cases. 

For a look at how to leverage W&B with PyTorch to debug your models through gradient tracking and visualizations, check out the [Debugging Neural Networks with PyTorch and W&B Using Gradients and Visualizations](https://wandb.ai/wandb_fc/articles/reports/Debugging-Neural-Networks-with-PyTorch-and-W-B-Using-Gradients-and-Visualizations--Vmlldzo1NDQxNTA5?galleryTag=reinforcement-learning) article.
