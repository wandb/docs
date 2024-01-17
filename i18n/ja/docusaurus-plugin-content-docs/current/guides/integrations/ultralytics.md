---
displayed_sidebar: ja
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Ultralytics

[**Ultralytics**](https://github.com/ultralytics/ultralytics) is the home for cutting-edge, state-of-the-art computer vision models for tasks like image classification, object detection, image segmentation, and pose estimation. Not only it hosts [**YOLOv8**](https://docs.ultralytics.com/models/yolov8/), the latest iteration in the **YOLO** series of real-time object detection models, but other powerful computer vision models such as [**SAM (Segment Anything Model)**](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [**RT-DETR**](https://docs.ultralytics.com/models/rtdetr/), [**YOLO-NAS**](https://docs.ultralytics.com/models/yolo-nas/), etc. Besides providing implementations of these models, Ultralytics also provides us with out-of-the-box workflows for training, fine-tuning, and applying these models using an easy-to-use API.

## Getting Started

First, we need to install `ultralytics`.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install ultralytics

# or
# conda install ultralytics
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install ultralytics
```

  </TabItem>
</Tabs>

Next, we need to install the [`feat/ultralytics`](https://github.com/wandb/wandb/tree/feat/ultralytics) branch from W&B, which currently houses the out-of-the-box integration for Ultralytics.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install git+https://github.com/wandb/wandb@feat/ultralytics
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install git+https://github.com/wandb/wandb@feat/ultralytics
```

  </TabItem>
</Tabs>

**Note:** The Ultralytcs integration will be soon available as a fully supported feature on Weights & Biases once [this pull request](https://github.com/wandb/wandb/pull/5867) is merged.

## Experiment Tracking and Visualizing Validation Results

This section demonstrates a typical workflow of using an [Ultralytics](https://docs.ultralytics.com/modes/predict/) model for training, fine-tuning, and validation and performing experiment tracking, model-checkpointing, and visualization of the model's performance using [Weights & Biases](https://wandb.ai/site).

You can try out the code in Google Colab: [Open In Colab](http://wandb.me/ultralytics-train)

You can also check out about the integration in this report: [**Supercharging Ultralytics with Weights & Biases**](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

In order to use the W&B integration with Ultralytics, we need to import the `wandb.yolov8.add_wandb_callback` function.

```python
import wandb
from wandb.yolov8 import add_wandb_callback

from ultralytics.engine.model import YOLO
```

Next, we initialize the `YOLO` model of our choice, and invoke the `add_wandb_callback` function on it before performing inference with the model. This would ensure that when we perform training, fine-tuning, validation, or inference, it would automatically log the experiment logs and the images over laid with both ground-truth and the respective prediction results using the [interactive overlays for computer vision tasks](../track/log/media#image-overlays-in-tables) on W&B along with additional insights in a [`wandb.Table`](../tables/intro.md).

```python
model_name = "yolov8n" #@param {type:"string"}
dataset_name = "coco128.yaml" #@param {type:"string"}

# Initialize YOLO Model
model = YOLO(f"{model_name}.pt")

# Add Weights & Biases callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train/fine-tune your model
# At the end of each epoch, predictions on validation batches are logged
# to a W&B table with insightful and interactive overlays for
# computer vision tasks
model.train(project="ultralytics", data=dataset_name, epochs=5, imgsz=640)
model.val()

# Finish the W&B run
wandb.finish()
```

Here's how experiments tracked using Weights & Biases for an Ultralytics training or fine-tuning workflow looks like:


Here's how epoch-wise validation results are visualized using a [Weights & Biases Table](../tables/intro.md):



## Visualizing Prediction Results

This section demonstrates a typical workflow of using an [Ultralytics](https://docs.ultralytics.com/modes/predict/) model for inference and visualizing the results using [Weights & Biases](https://wandb.ai/site).

You can try out the code in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/ultralytics-inference)

You can also check out about the integration in this report: [**Supercharging Ultralytics with Weights & Biases**](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

In order to use the W&B integration with Ultralytics, we need to import the `wandb.yolov8.add_wandb_callback` function.

```python
import wandb
from wandb.yolov8 import add_wandb_callback

from ultralytics.engine.model import YOLO
```

Now, let us download a few images to test the integration on. You can use your own images, videos or camera sources. For more information on inference sources, you can check out the [official docs](https://docs.ultralytics.com/modes/predict/).

```python
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

Next, we initialize a W&B [run](../runs/intro.md) using `wandb.init`.

```python
# Initialize Weights & Biases run
wandb.init(project="ultralytics", job_type="inference")
```

Next, we initialize the `YOLO` model of our choice, and invoke the `add_wandb_callback` function on it before performing inference with the model. This would ensure that when we perform inference, it would automatically log the images overlaid with our [interactive overlays for computer vision tasks](../track/log/media#image-overlays-in-tables) along with additional insights in a [`wandb.Table`](../tables/intro.md).

```python
model_name = 'yolov8n' #@param {type:"string"}

# Initialize YOLO Model
model = YOLO(f"{model_name}.pt")

# Add Weights & Biases callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Perform prediction which automatically logs to a W&B Table
# with interactive overlays for bounding boxes, segmentation masks
model(["./assets/img1.jpeg", "./assets/img3.png", "./assets/img4.jpeg", "./assets/img5.jpeg"])

# Finish the W&B run
wandb.finish()
```

**Note:** We do not need to explicitly initialize a run using `wandb.init()` in case of a training or fine-tuning workflow. However, tt is necessary to explicitly create a run, if the code only involves prediction.

Here's how the interactive bbox overlay looks:


You can fine more information on the W&B image overlays [here](../track/log/media.md#image-overlays).

