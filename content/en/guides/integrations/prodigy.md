---
description: How to integrate W&B with Prodigy.
menu:
  default:
    identifier: prodigy
    parent: integrations
title: Prodigy
weight: 290
---

[Prodigy](https://prodi.gy/) is an annotation tool for creating training and evaluation data for machine learning models, error analysis, data inspection & cleaning. [W&B Tables]({{< relref "/guides/models/tables/tables-walkthrough.md" >}}) allow you to log, visualize, analyze, and share datasets (and more!) inside W&B.

The [W&B integration with Prodigy](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) adds simple and easy-to-use functionality to upload your Prodigy-annotated dataset directly to W&B for use with Tables.

Run a few lines of code, like these:

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

and get visual, interactive, shareable tables like this one:

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="Prodigy annotation table" >}}

## Quickstart

Use `wandb.integration.prodigy.upload_dataset` to upload your annotated prodigy dataset directly from the local Prodigy database to W&B in our [Table]({{< relref "/ref/python/sdk/data-types/table.md" >}}) format. For more information on Prodigy, including installation & setup, please refer to the [Prodigy documentation](https://prodi.gy/docs/).

W&B will automatically try to convert images and named entity fields to [`wandb.Image`]({{< relref "/ref/python/sdk/data-types/image.md" >}}) and [`wandb.Html`]({{< relref "/ref/python/sdk/data-types/html.md" >}})respectively. Extra columns may be added to the resulting table to include these visualizations.

## Read through a detailed example

Explore the [Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) for example visualizations generated with W&B Prodigy integration.  

## Also using spaCy?

W&B also has an integration with spaCy, see the [docs here]({{< relref "/guides/integrations/spacy" >}}).