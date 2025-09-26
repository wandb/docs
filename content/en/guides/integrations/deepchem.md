---
description: How to integrate W&B with DeepChem library.
menu:
  default:
    identifier: deepchem
    parent: integrations
title: DeepChem
weight: 70
---
The [DeepChem library](https://github.com/deepchem/deepchem) provides open source tools that democratize the use of deep-learning in drug discovery, materials science, chemistry, and biology. This W&B integration adds simple and easy-to-use experiment tracking and model checkpointing while training models using DeepChem.

## DeepChem logging in 3 lines of code

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="DeepChem molecular analysis" >}}

## Report and Google Colab

Explore the Using [W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) article for an example charts generated using the W&B DeepChem integration.

To dive straight into working code, check out this [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb).

## Track experiments

Set up W&B for DeepChem models of type [KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) or [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models).

### Sign up and create an API key

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to the [W&B authorization page](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.

### Install the `wandb` library and log in

To install the `wandb` library locally and log in:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. Set the `WANDB_API_KEY` [environment variable]({{< relref "/guides/models/track/environment-variables.md" >}}) to your API key.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. Install the `wandb` library and log in.



    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}

{{< /tabpane >}}

### Log your training and evaluation data to W&B

Training loss and evaluation metrics can be automatically logged to W&B. Optional evaluation can be enabled using the DeepChem [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py), the `WandbLogger` will detect ValidationCallback callback and log the metrics generated.

{{< tabpane text=true >}}

{{% tab header="TorchModel" value="torch" %}}

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{% tab header="KerasModel" value="keras" %}}

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{< /tabpane >}}
