---
description: How to integrate W&B with PaddleOCR.
menu:
  default:
    identifier: paddleocr
    parent: integrations
title: PaddleOCR
weight: 280
---
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice implemented in PaddlePaddle. PaddleOCR support a variety of cutting-edge algorithms related to OCR, and developed industrial solution. PaddleOCR now comes with a Weights & Biases integration for logging training and evaluation metrics along with model checkpoints with corresponding metadata.

## Example Blog & Colab

[Read here](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw) to see how to train a model with PaddleOCR on the ICDAR2015 dataset. This also comes with a [Google Colab](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing) and the corresponding live W&B dashboard is available [here](https://wandb.ai/manan-goel/text_detection). There is also a Chinese version of this blog here: [W&B对您的OCR模型进行训练和调试](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## Sign up and create an API key

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to the [W&B authorization page](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.

## Install the `wandb` library and log in

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

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## Add wandb to your `config.yml` file

PaddleOCR requires configuration variables to be provided using a yaml file. Adding the following snippet at the end of the configuration yaml file will automatically log all training and validation metrics to a W&B dashboard along with model checkpoints:

```python
Global:
    use_wandb: True
```

Any additional, optional arguments that you might like to pass to [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) can also be added under the `wandb` header in the yaml file:

```
wandb:  
    project: CoolOCR  # (optional) this is the wandb project name 
    entity: my_team   # (optional) if you're using a wandb team, you can pass the team name here
    name: MyOCRModel  # (optional) this is the name of the wandb run
```

## Pass the `config.yml` file to `train.py`

The yaml file is then provided as an argument to the [training script](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py) available in the PaddleOCR repository.

```bash
python tools/train.py -c config.yml
```

Once you run your `train.py` file with Weights & Biases turned on, a link will be generated to bring you to your W&B dashboard:

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="PaddleOCR training dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="PaddleOCR validation dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="Text Detection Model dashboard" >}}

## Feedback or issues

If you have any feedback or issues about the Weights & Biases integration, open an issue on the [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) or email <a href="mailto:support@wandb.com">support@wandb.com</a>.