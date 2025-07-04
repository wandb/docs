---
description: How to Fine-Tune Azure OpenAI models using W&B.
menu:
  default:
    identifier: azure-openai-fine-tuning
    parent: integrations
title: Azure OpenAI Fine-Tuning
weight: 20
---

## Introduction
Fine-tuning GPT-3.5 or GPT-4 models on Microsoft Azure using W&B tracks, analyzes, and improves model performance by automatically capturing metrics and facilitating systematic evaluation through W&B's experiment tracking and evaluation tools.

{{< img src="/images/integrations/aoai_ft_plot.png" alt="Azure OpenAI fine-tuning metrics" >}}

## Prerequisites
- Set up Azure OpenAI service according to [official Azure documentation](https://wandb.me/aoai-wb-int).
- Configure a W&B account with an API key.

## Workflow overview

### 1. Fine-tuning setup
- Prepare training data according to Azure OpenAI requirements.
- Configure the fine-tuning job in Azure OpenAI.
- W&B automatically tracks the fine-tuning process, logging metrics and hyperparameters.

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

## Real-world example
* Explore the [medical note generation demo](https://wandb.me/aoai-ft-colab) to see how this integration facilitates:
  - Systematic tracking of fine-tuning experiments
  - Model evaluation using domain-specific metrics
* Go through an [interactive demo of fine-tuning a notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb)

## Additional resources
- [Azure OpenAI W&B Integration Guide](https://wandb.me/aoai-wb-int)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)