---
description: Tutorial on how to create sweep jobs from a pre-existing W&B project.
menu:
  default:
    identifier: existing-project
    parent: sweeps
title: 'Tutorial: Create sweep job from project'
---

This tutorial explains how to create sweep jobs from a pre-existing W&B project. We will use the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) to train a PyTorch convolutional neural network how to classify images. The required code an dataset is located in the [W&B examples repository (PyTorch CNN Fashion)](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

Explore the results in this [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion).

## 1. Create a project

First, create a baseline. Download the PyTorch MNIST dataset example model from W&B examples GitHub repository. Next, train the model. The training script is within the `examples/pytorch/pytorch-cnn-fashion` directory.

1. Clone this repo `git clone https://github.com/wandb/examples.git`
2. Open this example `cd examples/pytorch/pytorch-cnn-fashion`
3. Run a run manually `python train.py`

Optionally explore the example appear in the W&B App UI dashboard.

[View an example project page →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. Create a sweep

From your project page, open the [Sweep tab]({{< relref "./sweeps-ui.md" >}}) in the sidebar and select **Create Sweep**.

{{< img src="/images/sweeps/sweep1.png" alt="Sweep overview" >}}

The auto-generated configuration guesses values to sweep over based on the runs you have completed. Edit the configuration to specify what ranges of hyperparameters you want to try. When you launch the sweep, it starts a new process on the hosted W&B sweep server. This centralized service coordinates the agents— the machines that are running the training jobs.

{{< img src="/images/sweeps/sweep2.png" alt="Sweep configuration" >}}

## 3. Launch agents

Next, launch an agent locally. You can launch up to 20 agents on different machines in parallel if you want to distribute the work and finish the sweep job more quickly. The agent will print out the set of parameters it’s trying next.

{{< img src="/images/sweeps/sweep3.png" alt="Launch agents" >}}

Now you're running a sweep. The following image demonstrates what the dashboard looks like as the example sweep job is running. [View an example project page →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="Sweep dashboard" >}}

## Seed a new sweep with existing runs

Launch a new sweep using existing runs that you've previously logged.

1. Open your project table.
2. Select the runs you want to use with checkboxes on the left side of the table.
3. Click the dropdown to create a new sweep.

Your sweep will now be set up on our server. All you need to do is launch one or more agents to start running runs.

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="Seed sweep from runs" >}}

{{% alert %}}
If you kick off the new sweep as a bayesian sweep, the selected runs will also seed the Gaussian Process.
{{% /alert %}}