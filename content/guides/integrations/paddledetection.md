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

[**Read a W&B blog post**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) which illustrates how to integrate a YOLOX model with PaddleDetection on a subset of the `COCO2017` dataset.

## Use PaddleDetection with W&B

### Sign up and log in to W&B

[**Sign up**](https://wandb.ai/site) for a free Weights & Biases account, then pip install the wandb library. To login, you'll need to be signed in to you account at www.wandb.ai. Once signed in **you will find your API key on the** [**Authorize page**](https://wandb.ai/authorize)**.**

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```shell
pip install wandb

wandb login
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

```notebook
!pip install wandb

wandb.login()
```

{{% /tab %}}

{{< /tabpane >}}

### Activate the `WandbLogger` in your training script

#### Use the CLI

To use wandb via arguments to `train.py` in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/):

* Add the `--use_wandb` flag
* The first wandb arguments must be preceded by `-o` (you only need to pass this once)
* Each individual wandb argument must contain the prefix `wandb-` . For example any argument to be passed to [`wandb.init`]({{< relref "/ref/python/init" >}}) would get the `wandb-` prefix

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```

#### Use a `config.yml` file

You can also activate wandb via the config file. Add the wandb arguments to the config.yml file under the wandb header like so:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

Once you run your `train.py` file with Weights & Biases turned on, a link will be generated to bring you to your W&B dashboard:

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A Weights & Biases Dashboard" >}}

## Feedback or issues

If you have any feedback or issues about the Weights & Biases integration please open an issue on the [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) or email <a href="mailto:support@wandb.com">support@wandb.com</a>.