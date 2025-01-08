---
menu:
  default:
    identifier: mmengine
    parent: integrations
title: MMEngine
weight: 210
---
MMEngine by [OpenMMLab](https://github.com/open-mmlab) is a foundational library for training deep learning models based on PyTorch. MMEngine implements a next-generation training architecture for the OpenMMLab algorithm library, providing a unified execution foundation for over 30 algorithm libraries within OpenMMLab. Its core components include the training engine, evaluation engine, and module management.

[Weights and Biases](https://wandb.ai/site) is directly integrated into MMEngine through a dedicated [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) that can be used to
- log training and evaluation metrics.
- log and manage experiment configs.
- log additional records such as graph, images, scalars, etc.

## Get started

Install `openmim` and `wandb`.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="script" %}}

``` bash
pip install -q -U openmim wandb
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

``` bash
!pip install -q -U openmim wandb
```

{{% /tab %}}
{{< /tabpane >}}

Next, install `mmengine` and `mmcv` using `mim`.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="script" %}}

``` bash
mim install -q mmengine mmcv
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

``` bash
!mim install -q mmengine mmcv
```

{{% /tab %}}
{{< /tabpane >}}

## Use the `WandbVisBackend` with MMEngine Runner

This section demonstrates a typical workflow using `WandbVisBackend` using [`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner).

1. Define a `visualizer` from a visualization config.

    ```python
    from mmengine.visualization import Visualizer

    # define the visualization configs
    visualization_cfg = dict(
        name="wandb_visualizer",
        vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(project="mmengine"),
            )
        ],
        save_dir="runs/wandb"
    )

    # get the visualizer from the visualization configs
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    You pass a dictionary of arguments for [W&B run initialization](/ref/python/init) input parameters to `init_kwargs`.
    {{% /alert %}}

2. Initialize a `runner` with the `visualizer`, and call `runner.train()`.

    ```python
    from mmengine.runner import Runner

    # build the mmengine Runner which is a training helper for PyTorch
    runner = Runner(
        model,
        work_dir='runs/gan/',
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=opt_wrapper_dict,
        visualizer=visualizer, # pass the visualizer
    )

    # start training
    runner.train()
    ```

## Use the `WandbVisBackend` with OpenMMLab computer vision libraries

The `WandbVisBackend` can also be used easily to track experiments with OpenMMLab computer vision libraries such as [MMDetection](https://mmdetection.readthedocs.io/).

```python
# inherit base configs from the default runtime configs
_base_ = ["../_base_/default_runtime.py"]

# Assign the `WandbVisBackend` config dictionary to the
# `vis_backends` of the `visualizer` from the base configs
_base_.visualizer.vis_backends = [
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'mmdet',
            'entity': 'geekyrakshit'
        },
    ),
]
```