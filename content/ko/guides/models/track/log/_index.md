---
title: 오브젝트 및 미디어 로깅
description: 메트릭, 비디오, 커스텀 플롯 등 다양한 항목들을 추적하세요
cascade:
- url: guides/track/log/:filename
menu:
  default:
    identifier: ko-guides-models-track-log-_index
    parent: experiments
url: guides/track/log
weight: 6
---

W&B Python SDK를 사용하면 메트릭, 미디어 또는 커스텀 오브젝트의 사전을 한 스텝에 로그할 수 있습니다. W&B는 각 스텝에서 key-value 쌍을 수집하여, 매번 `wandb.Run.log()`로 데이터를 로그할 때 하나의 통합된 사전에 저장합니다. 스크립트에서 로그된 데이터는 로컬 머신의 `wandb` 디렉토리에 저장된 후, W&B 클라우드 또는 [private server]({{< relref path="/guides/hosting/" lang="ko" >}})로 동기화됩니다.

{{% alert %}}
key-value 쌍은 각 스텝마다 같은 값을 전달하는 경우에만 하나의 통합 사전에 저장됩니다. 만약 `step`에 대해 다른 값을 로그하면, W&B는 수집한 모든 키와 값을 메모리에 기록합니다.
{{% /alert %}}

`wandb.Run.log()`를 호출할 때마다 기본적으로 새로운 `step`이 생성됩니다. W&B는 차트 및 패널을 만들 때 기본적으로 step을 x축으로 사용합니다. 필요에 따라 커스텀 x축을 지정하거나 커스텀 summary 메트릭을 사용할 수 있습니다. 더 자세한 내용은 [Customize log axes]({{< relref path="./customize-logging-axes.md" lang="ko" >}})를 참고하세요.

{{% alert color="secondary" %}}
`wandb.Run.log()`를 사용할 때 각 `step`의 연속적인 값(0, 1, 2 등)을 로그하세요. 특정 history step에 값을 쓸 수 없습니다. W&B는 오직 "현재" 및 "다음" step에만 데이터를 기록합니다.
{{% /alert %}}


## 자동으로 로그되는 데이터

W&B는 W&B Experiment를 실행하는 동안 다음 정보를 자동으로 로그합니다.

* **System metrics**: CPU 및 GPU 사용률, 네트워크 등. GPU 메트릭의 경우 [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface)로 가져옵니다.
* **Command line**: stdout 및 stderr가 기록되어 [run 페이지]({{< relref path="/guides/models/track/runs/" lang="ko" >}})의 logs 탭에 표시됩니다.

계정의 [Settings page](https://wandb.ai/settings)에서 [Code Saving](https://wandb.me/code-save-colab)을 활성화하면 다음 항목도 기록됩니다:

* **Git commit**: 최신 git 커밋 정보가 run 페이지의 Overview 탭에 표시되며, 커밋하지 않은 변경사항이 있으면 `diff.patch` 파일도 확인할 수 있습니다.
* **Dependencies**: `requirements.txt` 파일이 업로드되어 run 페이지의 files 탭에 표시되며, run을 위해 `wandb` 디렉토리에 저장한 파일도 함께 확인할 수 있습니다.


## 특정 W&B API 호출에서 로그되는 데이터는?

W&B에서는 로그할 데이터를 직접 지정할 수 있습니다. 아래는 자주 로그되는 오브젝트들입니다:

* **Datasets**: 이미지 등 데이터셋 샘플은 직접 로그해야 W&B로 스트리밍됩니다.
* **Plots**: `wandb.plot()`을 `wandb.Run.log()`와 함께 사용해 차트를 추적할 수 있습니다. 자세한 내용은 [Log Plots]({{< relref path="./plots.md" lang="ko" >}})를 참고하세요.
* **Tables**: `wandb.Table`로 데이터를 로그하면 W&B에서 시각화 및 쿼리가 가능합니다. 자세한 내용은 [Log Tables]({{< relref path="./log-tables.md" lang="ko" >}})에서 확인하세요.
* **PyTorch gradients**: `wandb.Run.watch(model)`을 추가하면 UI에서 weight의 그레이디언트를 히스토그램으로 볼 수 있습니다.
* **설정 정보**: 하이퍼파라미터, 데이터셋 링크, 사용하는 아키텍처 이름 등을 config 파라미터로 로그할 수 있습니다. 예: `wandb.init(config=your_config_dictionary)`. 자세한 내용은 [PyTorch Integrations]({{< relref path="/guides/integrations/pytorch.md" lang="ko" >}}) 페이지를 참고하세요.
* **Metrics**: `wandb.Run.log()`를 사용해 모델의 메트릭을 확인할 수 있습니다. 트레이닝 루프에서 정확도, 손실 등 메트릭을 로그하면 UI에서 실시간 그래프를 볼 수 있습니다.


## 일반적인 워크플로우

1. **최고 정확도 비교**: 여러 run의 특정 메트릭 중 최고 값을 비교하려면 해당 메트릭의 summary 값을 설정합니다. 기본적으로 summary는 각 키의 마지막으로 로그된 값입니다. 이는 UI 표에서 summary 메트릭을 기준으로 run을 정렬하거나 필터링할 수 있으므로, 최종값이 아닌 _최고_ 정확도를 기준으로 run을 표 또는 바 차트상에서 쉽게 비교할 수 있습니다. 예: `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **여러 메트릭을 하나의 차트에서 보기**: `wandb.Run.log()`를 한 번 호출할 때 여러 메트릭을 함께 로그하면(예: `wandb.log({"acc": 0.9, "loss": 0.1})`), 두 항목 모두 UI에서 시각화 축으로 사용할 수 있습니다.
3. **x축 커스터마이즈**: 같은 로그 호출에 커스텀 x축 데이터를 추가하면, W&B 대시보드에서 다른 축을 기준으로 메트릭을 나타낼 수 있습니다. 예: `wandb.Run.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`. 특정 메트릭의 기본 x축을 지정하려면 [Run.define_metric()]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ko" >}})를 사용하세요.
4. **리치 미디어 및 차트 로그**: `wandb.Run.log()`는 [이미지, 비디오 등 미디어]({{< relref path="./media.md" lang="ko" >}})부터 [tables]({{< relref path="./log-tables.md" lang="ko" >}}), [charts]({{< relref path="/guides/models/app/features/custom-charts/" lang="ko" >}})까지 다양한 데이터 타입의 로그를 지원합니다.

## 모범 사례 및 팁

Experiments 및 로깅 관련 모범 사례와 꿀팁은 [Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging)에서 확인하실 수 있습니다.