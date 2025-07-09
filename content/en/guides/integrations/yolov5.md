---
menu:
  default:
    identifier: yolov5
    parent: integrations
title: YOLOv5
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics' YOLOv5](https://ultralytics.com/yolo) ("You Only Look Once") model family enables real-time object detection with convolutional neural networks without all the agonizing pain.

[Weights & Biases](http://wandb.com) is directly integrated into YOLOv5, providing experiment metric tracking, model and dataset versioning, rich model prediction visualization, and more. **It's as easy as running a single `pip install` before you run your YOLO experiments.**

{{% alert %}}
All W&B logging features are compatible with data-parallel multi-GPU training, such as with [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
{{% /alert %}}

## Track core experiments
Simply by installing `wandb`, you'll activate the built-in W&B [logging features]({{< relref "/guides/models/track/log/" >}}): system metrics, model metrics, and media logged to interactive [Dashboards]({{< relref "/guides/models/track/workspaces.md" >}}).

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # train a small network on a small dataset
```

Just follow the links printed to the standard out by wandb.

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="All these charts and more." >}}

## Customize the integration

By passing a few simple command line arguments to YOLO, you can take advantage of even more W&B features.

* If you pass a number to `--save_period`, W&B saves a [model version]({{< relref "/guides/core/registry/" >}}) at the end of every `save_period` epochs. The model version includes the model weights and tags the best-performing model in the validation set.
* Turning on the `--upload_dataset` flag will also upload the dataset for data versioning.
* Passing a number to `--bbox_interval` will turn on [data visualization]({{< relref "../" >}}). At the end of every `bbox_interval` epochs, the outputs of the model on the validation set will be uploaded to W&B.

{{< tabpane text=true >}}
{{% tab header="Model Versioning Only" value="modelversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1
```

{{% /tab %}}
{{% tab header="Model Versioning and Data Visualization" value="bothversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
Every W&B account comes with 100 GB of free storage for datasets and models.
{{% /alert %}}

Here's what that looks like.

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="Model versioning" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="Data visualization" >}}

{{% alert %}}
With data and model versioning, you can resume paused or crashed experiments from any device, no setup necessary. Check out [the Colab ](https://wandb.me/yolo-colab) for details.
{{% /alert %}}