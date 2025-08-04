---
description: How to integrate W&B with PaddleDetection.
menu:
  default:
    identifier: paddledetection
    parent: integrations
title: PaddleDetection
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) is an end-to-end object-detection development kit based on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle). It detects various mainstream objects, segments instances, and tracks and detects keypoints using configurable modules such as network components, data augmentations, and losses.

PaddleDetection now includes a built-in W&B integration which logs all your training and validation metrics, as well as your model checkpoints and their corresponding metadata.

The PaddleDetection `WandbLogger` logs your training and evaluation metrics to Weights & Biases as well as your model checkpoints while training.

[Read a W&B blog post](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) which illustrates how to integrate a YOLOX model with PaddleDetection on a subset of the `COCO2017` dataset.

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

{{% tab header="Python notebook" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## Activate the `WandbLogger` in your training script

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}
To use wandb via arguments to `train.py` in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/):

* Add the `--use_wandb` flag
* The first wandb arguments must be preceded by `-o` (you only need to pass this once)
* Each individual argument must contain the prefix `"wandb-"` . For example any argument to be passed to [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) would get the `wandb-` prefix

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```
{{% /tab %}}
{{% tab header="`config.yml`" value="config" %}}
Add the wandb arguments to the config.yml file under the `wandb` key:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

When you run your `train.py` file, it generates a link to your W&B dashboard.

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A Weights & Biases Dashboard" >}}
{{% /tab %}}
{{< /tabpane >}}

## Feedback or issues

If you have any feedback or issues about the Weights & Biases integration, open an issue on the [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) or email <a href="mailto:support@wandb.com">support@wandb.com</a>.