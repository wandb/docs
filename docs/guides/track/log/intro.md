---
title: Log media and objects in experiments
description: 메트릭, 비디오, 사용자 정의 플롯 등 다양한 항목을 추적하세요.
slug: /guides/track/log
displayed_sidebar: default
---

데이터, 미디어 또는 커스텀 오브젝트의 사전을 W&B Python SDK를 사용하여 `step`에 로그하세요. W&B는 각 `step`마다 키-값 쌍을 모아 로그할 때마다 하나의 통합된 사전에 저장합니다. 스크립트에서 로그된 데이터는 `wandb`라는 디렉토리에 로컬로 저장된 후, W&B 클라우드 또는 [private server](../../hosting/intro.md)로 동기화됩니다.

:::안내
키-값 쌍은 각 `step`마다 동일한 값을 전달할 경우에만 하나의 통합된 사전에 저장됩니다. `step`에 대해 다른 값을 로그할 경우 W&B는 수집된 모든 키와 값을 메모리에 저장합니다.
:::

각 `wandb.log` 호출은 기본적으로 새로운 `step`입니다. W&B는 차트와 패널을 생성할 때 기본 x축으로 `step`를 사용합니다. 필요에 따라 사용자 정의 x축을 생성하여 사용할 수 있으며, 사용자 정의 요약 메트릭을 캡처할 수 있습니다. 자세한 내용은 [Customize log axes](./customize-logging-axes.md)를 참조하세요.

:::caution
각 `step`에 대해 연속적인 값을 로그하려면 `wandb.log()`를 사용하세요: 0, 1, 2, 등등. 특정 히스토리 `step`에 쓰는 것은 불가능합니다. W&B는 "current"와 "next" `step`에만 기록합니다.
:::

## 자동으로 로그되는 데이터

W&B는 W&B Experiment 중 다음 정보를 자동으로 로그합니다:

- **System metrics**: CPU와 GPU 사용률, 네트워크 등. 이는 [run 페이지](../../app/pages/run-page.md)의 System 탭에 표시됩니다. GPU의 경우 [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface)를 사용해 가져옵니다.
- **Command line**: stdout과 stderr이 캡처되어 [run 페이지](../../app/pages/run-page.md)의 로그 탭에 표시됩니다.

계정의 [Settings 페이지](https://wandb.ai/settings)에서 [Code Saving](http://wandb.me/code-save-colab)을 활성화하여 다음을 로그하세요:

- **Git commit**: 최신 git 커밋을 수집하여 run 페이지의 overview 탭에서 확인할 수 있으며, 커밋되지 않은 변경 사항이 있는 경우 `diff.patch` 파일도 함께 확인할 수 있습니다.
- **Dependencies**: `requirements.txt` 파일이 업로드되어 run 페이지의 파일 탭에 표시되며, `wandb` 디렉토리에 저장한 파일과 함께 표시됩니다.

## 특정 W&B API 호출로 로그되는 데이터는 무엇입니까?

W&B를 사용하면 로그하고 싶은 것을 정확히 결정할 수 있습니다. 다음은 일반적으로 로그되는 오브젝트 목록입니다:

- **Datasets**: 이미지 또는 다른 데이터셋 샘플을 W&B로 스트리밍하려면 명시적으로 로그해야 합니다.
- **Plots**: `wandb.plot`을 `wandb.log`와 함께 사용하여 차트를 추적합니다. 자세한 내용은 [Log Plots](./plots.md)를 참조하세요.
- **Tables**: `wandb.Table`을 사용하여 데이터를 로그하고 W&B를 통해 시각화하고 쿼리합니다. 자세한 내용은 [Log Tables](./log-tables.md)를 참조하세요.
- **PyTorch gradients**: `wandb.watch(model)`을 추가하여 UI에서 히스토그램으로 가중치의 그레이디언트를 확인합니다.
- **Configuration information**: 하이퍼파라미터, 데이터셋 링크 또는 사용 중인 아키텍처 이름을 설정 파라미터로 로그하고, 다음과 같이 전달합니다: `wandb.init(config=your_config_dictionary)`. 자세한 내용은 [PyTorch Integrations](../../integrations/pytorch.md) 페이지를 참조하세요. 
- **Metrics**: `wandb.log`를 사용하여 모델의 메트릭을 확인합니다. 트레이닝 루프 내에서 정확도나 손실 같은 메트릭을 로그하면 UI에서 실시간으로 업데이트되는 그래프를 얻을 수 있습니다.

## 일반적인 워크플로우

1. **최고 정확도 비교**: `runs` 사이 메트릭의 최고 값을 비교하려면 그 메트릭의 summary 값을 설정하십시오. 기본적으로 summary는 각 키에 대해 마지막으로 로그된 값으로 설정됩니다. 이는 UI의 테이블에서 `runs`를 정렬하고 필터링할 수 있는 데 유용합니다. 따라서 최종 정확도 대신 _최고_ 정확도를 기준으로 `runs`를 테이블이나 막대 차트로 비교할 수 있습니다. 예를 들어, summary를 다음과 같이 설정할 수 있습니다: `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **한 차트에 여러 메트릭 로그**: 같은 `wandb.log` 호출에서 여러 메트릭을 로그하세요, 예를 들면: `wandb.log({"acc'": 0.9, "loss": 0.1})`. 그러면 둘 다 UI에서 플롯할 수 있습니다.
3. **커스텀 x축**: 동일한 로그 호출에 커스텀 x축을 추가하여 W&B 대시보드에서 다른 축에 대하여 메트릭을 시각화합니다. 예: `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`. 주어진 메트릭의 기본 x축을 설정하려면 [Run.define_metric()](../../../ref/python/run.md#define_metric)를 사용하세요.
4. **리치 미디어 및 차트 로그**: `wandb.log`는 이미지나 비디오 같은 [미디어](./media.md)에서 [테이블](./log-tables.md) 및 [차트](../../app/features/custom-charts/intro.md)에 이르기까지 다양한 데이터 유형의 로그를 지원합니다.