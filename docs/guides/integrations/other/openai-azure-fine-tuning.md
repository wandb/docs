---
slug: /guides/integrations/openai-azure-fine-tuning
description: How to Fine-Tune Azure OpenAI models using W&B.
displayed_sidebar: default
---

# Azure OpenAI Fine-Tuning

## Introduction
Fine-tuning GPT-3.5 or GPT-4 models on Microsoft Azure using Weights & Biases allows for detailed tracking and analysis of model performance. This guide extends the concepts from the [OpenAI Fine-Tuning guide](/guides/integrations/openai) with specific steps and features for Azure OpenAI.

![](/images/integrations/open_ai_auto_scan.png)

:::info
The Weights and Biases fine-tuning integration works with `openai >= 1.0`. Please install the latest version of `openai` by doing `pip install -U openai`.
:::

### Check out interactive examples

* [Demo Colab](http://wandb.me/azure-openai-colab)

## Prerequisites
- Azure OpenAI service set up as per [official Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune).
- Latest versions of `openai`, `wandb`, and other required libraries installed.

## Sync Azure OpenAI Fine-Tuning Results in Weights & Biases
### Setting Up
- Ensure you have the Azure OpenAI endpoint and key configured in your environment.

```python
import os

os.environ["AZURE_OPENAI_ENDPOINT"] = None  # Replace with your endpoint
os.environ["AZURE_OPENAI_KEY"] = None  # Replace with your key
```

- Install necessary libraries (`openai`, `requests`, `tiktoken`, `wandb`).

```shell-session
pip install openai requests tiktoken wandb
```

### Preparing Datasets
- Create and validate your training and validation datasets in JSONL format, as demonstrated in the example datasets provided in the guide.

```python
# Example dataset creation
# %%writefile training_set.jsonl
# {"messages": [{"role": "system", "content": "Clippy is a factual chatbot that is also sarcastic."}, ...]}
```

### Fine-Tuning on Azure

1. **Upload Datasets:** Use Azure OpenAI SDK to upload your training and validation datasets.

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)
training_response = client.files.create(
    file=open("training_set.jsonl", "rb"), purpose="fine-tune"
)
training_file_id = training_response.id
```

2. **Start Fine-Tuning:** Initiate the fine-tuning process on Azure with the desired base model, like `gpt-35-turbo-0613`.

```python
response = client.fine_tuning.jobs.create(
    training_file=training_file_id, model="gpt-35-turbo-0613"
)
job_id = response.id
```

3. **Track the Fine-Tuning Job: Integrating with Weights & Biases**
- Use the `WandbLogger` from `wandb.integration.openai.fine_tuning` just as in the [OpenAI Fine-Tuning guide](/guides/integrations/openai).
- The `WandbLogger.sync` method takes the fine-tune job ID and other optional parameters to sync your fine-tuning results to Weights & Biases.
- This integration will log training/validation metrics, datasets, model metadata, and establish data and model DAG lineage in Weights & Biases.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

WandbLogger.sync(
    fine_tune_job_id=job_id, openai_client=client, project="your_project_name"
)
```

## Visualization and Versioning in Weights & Biases
- Utilize Weights & Biases for versioning and visualizing training and validation data as Tables.
- The datasets and model metadata are versioned as W&B Artifacts, allowing for efficient tracking and version control.

![](/images/integrations/openai_data_artifacts.png)

![](/images/integrations/openai_data_visualization.png)

## Retrieving the Fine-Tuned Model
- The fine-tuned model ID is retrievable from Azure OpenAI and is logged as a part of model metadata in Weights & Biases.

![](/images/integrations/openai_model_metadata.png)

## Additional Resources
- [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/)
- [Demo Colab](http://wandb.me/azure-openai-colab)