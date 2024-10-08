---
title: MMEngine
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

MMEngine by [OpenMMLab](https://github.com/open-mmlab)은 PyTorch 기반의 딥러닝 모델 트레이닝을 위한 기초 라이브러리입니다. MMEngine은 OpenMMLab 알고리즘 라이브러리를 위한 차세대 트레이닝 아키텍처를 구현하며, OpenMMLab 내 30개 이상의 알고리즘 라이브러리들에 대한 통합 실행 기반을 제공합니다. 주요 구성 요소로는 트레이닝 엔진, 평가 엔진, 모듈 관리가 포함됩니다.

[Weights and Biases](https://wandb.ai/site)는 `WandbVisBackend` [(API 링크)](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) 를 통해 MMEngine에 직접 통합됩니다. 이를 통해
- 트레이닝 및 평가 메트릭을 로그할 수 있습니다.
- 실험 설정을 로그하고 관리할 수 있습니다.
- 그래프, 이미지, 스칼라 등 추가적인 기록을 로그할 수 있습니다.

## 시작하기

먼저 `openmim`과 `wandb`를 설치해야 합니다. 그런 다음, `openmim`을 사용하여 `mmengine` 및 `mmcv`를 설치할 수 있습니다.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install -q -U openmim wandb
mim install -q mmengine mmcv
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install -q -U openmim wandb
!mim install -q mmengine mmcv
```

  </TabItem>
</Tabs>

## MMEngine Runner에서 `WandbVisBackend` 사용하기

이 섹션에서는 [`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner)를 사용하여 `WandbVisBackend`를 사용하는 일반적인 워크플로우를 보여줍니다.

먼저, 시각화 설정에서 `visualizer`를 정의해야 합니다.

```python
from mmengine.visualization import Visualizer

# 시각화 설정을 정의합니다.
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

# 시각화 설정에서 visualizer를 가져옵니다.
visualizer = Visualizer.get_instance(**visualization_cfg)
```

:::info
[W&B run 초기화](/ref/python/init) 인수의 입력 파라미터를 `init_kwargs`에 사전 형태로 전달합니다.
:::

다음으로, `visualizer`와 함께 `runner`를 초기화하고 `runner.train()`을 호출하면 됩니다.

```python
from mmengine.runner import Runner

# PyTorch를 위한 트레이닝 도우미인 mmengine Runner를 구축합니다.
runner = Runner(
    model,
    work_dir='runs/gan/',
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict,
    visualizer=visualizer, # visualizer를 전달합니다.
)

# 트레이닝을 시작합니다.
runner.train()
```

| ![An example of your experiment tracked using the `WandbVisBackend`](/images/integrations/mmengine.png) | 
|:--:| 
| **`WandbVisBackend`을 사용하여 실험을 추적한 예시입니다.** |

## OpenMMLab 컴퓨터 비전 라이브러리와 함께 `WandbVisBackend` 사용하기

`WandbVisBackend`는 [MMDetection](https://mmdetection.readthedocs.io/)과 같은 OpenMMLab 컴퓨터 비전 라이브러리와의 실험을 추적하는 데에도 쉽게 사용할 수 있습니다.

```python
# 기본 런타임 설정에서 기본 설정을 상속받습니다.
_base_ = ["../_base_/default_runtime.py"]

# `visualizer`에서 `vis_backends`에 `WandbVisBackend` 설정 사전을 할당합니다.
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