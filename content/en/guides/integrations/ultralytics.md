---
menu:
  default:
    identifier: ultralytics
    parent: integrations
title: Ultralytics
weight: 480
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics) is the home for cutting-edge, state-of-the-art computer vision models for tasks like image classification, object detection, image segmentation, and pose estimation. Not only it hosts [YOLOv8](https://docs.ultralytics.com/models/yolov8/), the latest iteration in the YOLO series of real-time object detection models, but other powerful computer vision models such as [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/), etc. Besides providing implementations of these models, Ultralytics also provides us with out-of-the-box workflows for training, fine-tuning, and applying these models using an easy-to-use API.

## Get started

1. Install `ultralytics` and `wandb`.

    {{< tabpane text=true >}}
    {{% tab header="Command Line" value="script" %}}

    ```shell
    pip install --upgrade ultralytics==8.0.238 wandb

    # or
    # conda install ultralytics
    ```

    {{% /tab %}}
    {{% tab header="Notebook" value="notebook" %}}

    ```bash
    !pip install --upgrade ultralytics==8.0.238 wandb
    ```

    {{% /tab %}}
    {{< /tabpane >}}

    The development team has tested the integration with `ultralyticsv8.0.238` and below. To report any issues with the integration, create a [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml) with the tag `yolov8`.

## Track experiments and visualize validation results

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

This section demonstrates a typical workflow of using an [Ultralytics](https://docs.ultralytics.com/modes/predict/) model for training, fine-tuning, and validation and performing experiment tracking, model-checkpointing, and visualization of the model's performance using [W&B](https://wandb.ai/site).

You can also check out about the integration in this report: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

To use the W&B integration with Ultralytics, import the `wandb.integration.ultralytics.add_wandb_callback` function.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

Initialize the `YOLO` model of your choice, and invoke the `add_wandb_callback` function on it before performing inference with the model. This ensures that when you perform training, fine-tuning, validation, or inference, it automatically saves the experiment logs and the images, overlaid with both ground-truth and the respective prediction results using the [interactive overlays for computer vision tasks]({{< relref "/guides/models/track/log/media#image-overlays-in-tables" >}}) on W&B along with additional insights in a [`wandb.Table`]({{< relref "/guides/models/tables/" >}}).

```python
with wandb.init(project="ultralytics", job_type="train") as run:

    # Initialize YOLO Model
    model = YOLO("yolov8n.pt")

    # Add W&B callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Train/fine-tune your model
    # At the end of each epoch, predictions on validation batches are logged
    # to a W&B table with insightful and interactive overlays for
    # computer vision tasks
    model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)
```

Here's how experiments tracked using W&B for an Ultralytics training or fine-tuning workflow looks like:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

Here's how epoch-wise validation results are visualized using a [W&B Table]({{< relref "/guides/models/tables/" >}}):

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## Visualize prediction results

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

This section demonstrates a typical workflow of using an [Ultralytics](https://docs.ultralytics.com/modes/predict/) model for inference and visualizing the results using [W&B](https://wandb.ai/site).

You can try out the code in Google Colab: [Open in Colab](https://wandb.me/ultralytics-inference).

You can also check out about the integration in this report: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

In order to use the W&B integration with Ultralytics, we need to import the `wandb.integration.ultralytics.add_wandb_callback` function.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

Download a few images to test the integration on. You can use still images, videos, or camera sources. For more information on inference sources, check out the [Ultralytics docs](https://docs.ultralytics.com/modes/predict/).

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

Initialize a W&B [run]({{< relref "/guides/models/track/runs/" >}}) using `wandb.init()`. Next, Initialize your desired `YOLO` model and invoke the `add_wandb_callback` function on it before you perform inference with the model. This ensures that when you perform inference, it automatically logs the images overlaid with your [interactive overlays for computer vision tasks]({{< relref "/guides/models/track/log/media#image-overlays-in-tables" >}}) along with additional insights in a [`wandb.Table`]({{< relref "/guides/models/tables/" >}}).

```python
# Initialize W&B run
with wandb.init(project="ultralytics", job_type="inference") as run:
    # Initialize YOLO Model
    model = YOLO("yolov8n.pt")

    # Add W&B callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Perform prediction which automatically logs to a W&B Table
    # with interactive overlays for bounding boxes, segmentation masks
    model(
        [
            "./assets/img1.jpeg",
            "./assets/img3.png",
            "./assets/img4.jpeg",
            "./assets/img5.jpeg",
        ]
    )
```

You do not need to explicitly initialize a run using `wandb.init()` in case of a training or fine-tuning workflow. However, if the code involves only prediction, you must explicitly create a run.

Here's how the interactive bbox overlay looks:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

For more details, see the [W&B image overlays guide]({{< relref "/guides/models/track/log/media.md#image-overlays" >}}).

## More resources

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)