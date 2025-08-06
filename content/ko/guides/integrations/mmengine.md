---
title: 'MMEngine


  '
menu:
  default:
    identifier: ko-guides-integrations-mmengine
    parent: integrations
weight: 210
---

MMEngine은 [OpenMMLab](https://github.com/open-mmlab) 에서 개발한 파이토치 기반 딥러닝 모델 트레이닝의 핵심 라이브러리입니다. MMEngine은 OpenMMLab 알고리즘 라이브러리를 위한 차세대 트레이닝 아키텍처를 제공하며, OpenMMLab의 30개가 넘는 알고리즘 라이브러리에 공통 실행 기반을 마련합니다. 주요 구성 요소에는 트레이닝 엔진, 평가 엔진, 모듈 관리 등이 있습니다.

[W&B](https://wandb.ai/site)는 MMEngine에 전용 [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) 로 직접 통합되어 있으며, 아래와 같은 작업에 활용할 수 있습니다.
- 트레이닝 및 평가 메트릭 기록
- 실험 config 관리 및 기록
- 그래프, 이미지, 스칼라 등 추가 기록

## 시작하기

`openmim` 과 `wandb` 를 설치하세요.

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

다음으로 `mmengine` 과 `mmcv`를 `mim` 으로 설치하세요.

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

## MMEngine Runner에서 `WandbVisBackend` 사용하기

이 섹션에서는 [`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) 를 활용하여 `WandbVisBackend`를 사용하는 대표적인 워크플로우를 소개합니다.

1. 시각화 config에서 `visualizer` 정의

    ```python
    from mmengine.visualization import Visualizer

    # 시각화 config를 정의합니다
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

    # 시각화 config에서 visualizer를 가져옵니다
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    [W&B run 초기화]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) ARG들은 key-value 딕셔너리 형식으로 `init_kwargs`에 전달할 수 있습니다.
    {{% /alert %}}

2. `visualizer`로 `runner`를 초기화한 후, `runner.train()`을 호출합니다.

    ```python
    from mmengine.runner import Runner

    # PyTorch 트레이닝 헬퍼인 mmengine Runner를 만듭니다
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

## OpenMMLab 컴퓨터 비전 라이브러리에서 `WandbVisBackend` 사용하기

`WandbVisBackend`는 [MMDetection](https://mmdetection.readthedocs.io/) 등 OpenMMLab 컴퓨터 비전 라이브러리에서도 간단하게 실험 추적에 사용할 수 있습니다.

```python
# 기본 런타임 config에서 base config를 상속합니다
_base_ = ["../_base_/default_runtime.py"]

# `WandbVisBackend` config 딕셔너리를
# base config의 visualizer의 `vis_backends`에 할당합니다
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