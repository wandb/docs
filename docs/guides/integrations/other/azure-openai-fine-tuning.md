---
slug: /guides/integrations/azure-openai-fine-tuning
description: How to Fine-Tune Azure OpenAI models using W&B.
displayed_sidebar: default
title: Azure OpenAI Fine-Tuning
---

## Introduction
Fine-tuning GPT-3.5 or GPT-4 models on Microsoft Azure using W&B allows you to track, analyze, and improve your model's performance by automatically capturing metrics and enabling systematic model evaluation through W&B's experiment tracking and evaluation tools.

![](/images/integrations/aoai_ft_plot.png)

## Prerequisites
- Azure OpenAI service set up as per [official Azure documentation](https://wandb.me/aoai-wb-int)
- W&B account with API key configured

## Workflow overview

### 1. Fine-tuning setup
- Prepare your training data according to Azure OpenAI requirements
- Configure your fine-tuning job in Azure OpenAI
- W&B automatically tracks the fine-tuning process, logging metrics and hyperparameters

### 2. Experiment tracking
During fine-tuning, W&B captures:
- Training and validation metrics
- Model hyperparameters
- Resource utilization
- Training artifacts

### 3. Model evaluation
After fine-tuning, use [W&B Weave](https://weave-docs.wandb.ai) to:
- Evaluate model outputs against reference datasets
- Compare performance across different fine-tuning runs
- Analyze model behavior on specific test cases
- Make data-driven decisions for model selection

## Real-World example
Check out the [medical note generation demo](https://wandb.me/aoai-ft-colab) to see how this integration enables:
- Systematic tracking of fine-tuning experiments
- Model evaluation using domain-specific metrics
- Version comparison for selecting the best model

### Interactive demo
* [Fine-tuning Notebook](https://wandb.me/aoai-med-ft)

## Additional resources
- [Azure OpenAI W&B Integration Guide](https://wandb.me/aoai-wb-int)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)