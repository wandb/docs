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

{{< img src="/images/integrations/cd.png" alt="" >}}

## Report and Google Colab

Explore the Using [W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) article for an example charts generated using the W&B DeepChem integration.

If you'd rather dive straight into working code, check out this [**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb).

## Track experiments

Setup Weights & Biases for DeepChem models of type [KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) or [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models).

1. Install the `wandb` library and log in

    {{< tabpane text=true >}}

    {{% tab header="Command Line" value="cli" %}}

    ```
    pip install wandb
    wandb login
    ```

    {{% /tab %}}

    {{% tab header="Notebook" value="notebook" %}}

    ```python
    !pip install wandb

    import wandb
    wandb.login()
    ```

    {{% /tab %}}

    {{< /tabpane >}}

2. Initialize and configure WandbLogger

    ```python
    from deepchem.models import WandbLogger

    logger = WandbLogger(entity="my_entity", project="my_project")
    ```

3. Log your training and evaluation data to W&B

    Training loss and evaluation metrics can be automatically logged to Weights & Biases. Optional evaluation can be enabled using the DeepChem [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py), the `WandbLogger` will detect ValidationCallback callback and log the metrics generated.

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
