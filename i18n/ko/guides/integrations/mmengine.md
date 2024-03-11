---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# MMEngine

[OpenMMLab](https://github.com/open-mmlab)의 MMEngine은 PyTorch 기반의 딥러닝 모델을 트레이닝하기 위한 기초 라이브러리입니다. MMEngine은 OpenMMLab 알고리즘 라이브러리에 대한 차세대 트레이닝 아키텍처를 구현하며, OpenMMLab 내 30개 이상의 알고리즘 라이브러리에 대한 통합 실행 기반을 제공합니다. 그 핵심 구성 요소로는 트레이닝 엔진, 평가 엔진, 그리고 모듈 관리가 포함됩니다.

[Weights and Biases](https://wandb.ai/site)는 [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend)를 통해 MMEngine에 직접 통합되어
- 트레이닝 및 평가 메트릭 로깅.
- 실험 설정 로깅 및 관리.
- 그래프, 이미지, 스칼라 등과 같은 추가적인 기록 로깅.

## 시작하기

먼저, `openmim`과 `wandb`를 설치해야 합니다. 그 다음 `openmim`을 사용하여 `mmengine`과 `mmcv`를 설치할 수 있습니다.

<Tabs
  defaultValue="script"
  values={[
    {label: '커맨드라인', value: 'script'},
    {label: '노트북', value: 'notebook'},
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

## MMEngine Runner와 `WandbVisBackend` 사용하기

이 섹션은 [`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner)를 사용하여 `WandbVisBackend`를 사용하는 전형적인 워크플로우를 보여줍니다.

먼저, 시각화 설정에서 `visualizer`를 정의해야 합니다.

```python
from mmengine.visualization import Visualizer

# 시각화 설정 정의
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

# 시각화 설정에서 visualizer 가져오기
visualizer = Visualizer.get_instance(**visualization_cfg)
```

:::info
`init_kwargs`에는 [W&B run 초기화](https://docs.wandb.ai/ref/python/init) 입력 파라미터에 대한 인수 사전을 전달합니다.
:::

다음으로, `visualizer`와 함께 `runner`를 초기화하고 `runner.train()`을 호출하면 됩니다.

```python
from mmengine.runner import Runner

# PyTorch에 대한 트레이닝 도우미인 mmengine Runner 구축
runner = Runner(
    model,
    work_dir='runs/gan/',
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict,
    visualizer=visualizer, # visualizer 전달
)

# 트레이닝 시작
runner.train()
```

| ![`WandbVisBackend`을 사용하여 추적된 실험의 예시](@site/static/images/integrations/mmengine.png) | 
|:--:| 
| **`WandbVisBackend`을 사용하여 추적된 실험의 예시.** |

## OpenMMLab 컴퓨터 비전 라이브러리와 `WandbVisBackend` 사용하기

`WandbVisBackend`는 [MMDetection](https://mmdetection.readthedocs.io/)과 같은 OpenMMLab 컴퓨터 비전 라이브러리에서도 실험을 추적하는 데 쉽게 사용될 수 있습니다.

```python
# 기본 런타임 설정에서 기본 설정 상속
_base_ = ["../_base_/default_runtime.py"]

# 기본 설정에서 `visualizer`의 `vis_backends`에 `WandbVisBackend` 설정 사전을 할당
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