---
description: How to integrate W&B with YOLOX.
menu:
  default:
    identifier: yolox
    parent: integrations
title: YOLOX
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) is an anchor-free version of YOLO with strong performance for object detection. You can use the YOLOX W&B integration to turn on logging of metrics related to training, validation, and the system, and you can interactively validate predictions with a single command-line argument.

## Sign up and create an API key

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to [https://wandb.ai/authorize](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
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

## Log metrics

Use the `--logger wandb` command line argument to turn on logging with wandb. Optionally you can also pass all of the arguments that [`wandb.init`]({{< relref "/ref/python/init" >}}) expects; prepend each argument with `wandb-`.

`num_eval_imges` controls the number of validation set images and predictions that are  logged to W&B tables for model evaluation.

```shell
# login to wandb
wandb login

# call your yolox training script with the `wandb` logger argument
python tools/train.py .... --logger wandb \
                wandb-project <project-name> \
                wandb-entity <entity>
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
```

## Example

[Example dashboard with YOLOX training and validation metrics ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="" >}}

Any questions or issues about this W&B integration? Open an issue in the [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX).