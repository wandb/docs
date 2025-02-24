---
title: MMEngine
menu:
  default:
    identifier: ko-guides-integrations-mmengine
    parent: integrations
weight: 210
---

OpenMMLab의 [MMEngine](https://github.com/open-mmlab)은 PyTorch를 기반으로 딥러닝 모델을 트레이닝하기 위한 기본 라이브러리입니다. MMEngine은 OpenMMLab 알고리즘 라이브러리를 위한 차세대 트레이닝 아키텍처를 구현하여 OpenMMLab 내의 30개 이상의 알고리즘 라이브러리에 대한 통합 실행 기반을 제공합니다. 핵심 구성 요소에는 트레이닝 엔진, 평가 엔진 및 모듈 관리가 포함됩니다.

[Weights and Biases](https://wandb.ai/site)는 [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend)를 통해 MMEngine에 직접 통합되어 다음을 수행하는 데 사용할 수 있습니다.
- 트레이닝 및 평가 메트릭 기록
- experiment 구성 기록 및 관리
- 그래프, 이미지, 스칼라 등과 같은 추가 레코드 기록

## 시작하기

`openmim` 및 `wandb`를 설치합니다.

{{< tabpane text=true >}}
{{% tab header="커맨드 라인" value="script" %}}

``` bash
pip install -q -U openmim wandb
```

{{% /tab %}}

{{% tab header="노트북" value="notebook" %}}

``` bash
!pip install -q -U openmim wandb
```

{{% /tab %}}
{{< /tabpane >}}

다음으로 `mim`을 사용하여 `mmengine` 및 `mmcv`를 설치합니다.

{{< tabpane text=true >}}
{{% tab header="커맨드 라인" value="script" %}}

``` bash
mim install -q mmengine mmcv
```

{{% /tab %}}

{{% tab header="노트북" value="notebook" %}}

``` bash
!mim install -q mmengine mmcv
```

{{% /tab %}}
{{< /tabpane >}}

## MMEngine Runner와 함께 `WandbVisBackend` 사용

이 섹션에서는 [`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner)를 사용하여 `WandbVisBackend`를 사용하는 일반적인 워크플로우를 보여줍니다.

1. 시각화 구성에서 `visualizer`를 정의합니다.

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
    [W&B run 초기화]({{< relref path="/ref/python/init" lang="ko" >}}) 입력 파라미터에 대한 인수의 사전을 `init_kwargs`에 전달합니다.
    {{% /alert %}}

2. `visualizer`로 `runner`를 초기화하고 `runner.train()`을 호출합니다.

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

## OpenMMLab 컴퓨터 비전 라이브러리와 함께 `WandbVisBackend` 사용

`WandbVisBackend`는 [MMDetection](https://mmdetection.readthedocs.io/)과 같은 OpenMMLab 컴퓨터 비전 라이브러리를 사용하여 Experiments를 쉽게 추적하는 데 사용할 수도 있습니다.

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
