---
title: Log objects and media
description: 메트릭, 비디오, 사용자 지정 플롯 등을 추적하세요.
cascade:
- url: guides/track/log/:filename
menu:
  default:
    identifier: ko-guides-models-track-log-_index
    parent: experiments
url: guides/track/log
weight: 6
---

W&B Python SDK를 사용하여 메트릭, 미디어 또는 사용자 정의 오브젝트의 사전을 step에 기록합니다. W&B는 각 step 동안 키-값 쌍을 수집하고 `wandb.log()`로 데이터를 기록할 때마다 하나의 통합된 사전에 저장합니다. 스크립트에서 기록된 데이터는 로컬 시스템의 `wandb`라는 디렉토리에 저장된 다음, W&B cloud 또는 [개인 서버]({{< relref path="/guides/hosting/" lang="ko" >}})에 동기화됩니다.

{{% alert %}}
키-값 쌍은 각 step에 대해 동일한 값을 전달하는 경우에만 하나의 통합된 사전에 저장됩니다. `step`에 대해 다른 값을 기록하면 W&B는 수집된 모든 키와 값을 메모리에 씁니다.
{{% /alert %}}

`wandb.log`를 호출할 때마다 기본적으로 새로운 `step`이 됩니다. W&B는 차트와 패널을 만들 때 step을 기본 x축으로 사용합니다. 선택적으로 사용자 정의 x축을 만들고 사용할 수도 있고, 사용자 정의 요약 메트릭을 캡처할 수도 있습니다. 자세한 내용은 [로그 축 사용자 정의]({{< relref path="./customize-logging-axes.md" lang="ko" >}})를 참조하십시오.

{{% alert color="secondary" %}}
각 `step`에 대해 연속적인 값(0, 1, 2 등)을 기록하려면 `wandb.log()`를 사용하십시오. 특정 history step에 쓰는 것은 불가능합니다. W&B는 "현재" 및 "다음" step에만 씁니다.
{{% /alert %}}

## 자동으로 기록되는 데이터

W&B는 W&B Experiments 동안 다음 정보를 자동으로 기록합니다.

* **시스템 메트릭**: CPU 및 GPU 사용률, 네트워크 등. 이러한 정보는 [run 페이지]({{< relref path="/guides/models/track/runs/" lang="ko" >}})의 System 탭에 표시됩니다. GPU의 경우, 이러한 정보는 [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface)를 통해 가져옵니다.
* **커맨드라인**: stdout 및 stderr이 선택되어 [run 페이지]({{< relref path="/guides/models/track/runs/" lang="ko" >}})의 로그 탭에 표시됩니다.

계정의 [설정 페이지](https://wandb.ai/settings)에서 [코드 저장](http://wandb.me/code-save-colab)을 켜면 다음을 기록할 수 있습니다.

* **Git 커밋**: 최신 git 커밋을 선택하여 run 페이지의 Overview 탭에서 확인할 수 있으며, 커밋되지 않은 변경 사항이 있는 경우 `diff.patch` 파일도 확인할 수 있습니다.
* **Dependencies**: `requirements.txt` 파일이 업로드되어 run 페이지의 파일 탭에 표시되며, run을 위해 `wandb` 디렉토리에 저장하는 모든 파일과 함께 표시됩니다.

## 특정 W&B API 호출로 기록되는 데이터는 무엇입니까?

W&B를 사용하면 기록할 내용을 정확히 결정할 수 있습니다. 다음은 일반적으로 기록되는 오브젝트의 일부 목록입니다.

* **Datasets**: 이미지를 포함한 데이터셋 샘플을 W&B로 스트리밍하려면 명시적으로 기록해야 합니다.
* **Plots**: 차트를 추적하려면 `wandb.plot`을 `wandb.log`와 함께 사용하십시오. 자세한 내용은 [Plots 기록]({{< relref path="./plots.md" lang="ko" >}})을 참조하십시오.
* **Tables**: `wandb.Table`을 사용하여 W&B로 시각화하고 쿼리할 데이터를 기록합니다. 자세한 내용은 [Tables 기록]({{< relref path="./log-tables.md" lang="ko" >}})을 참조하십시오.
* **PyTorch gradients**: `wandb.watch(model)`을 추가하여 가중치의 그래디언트를 UI에서 히스토그램으로 확인합니다.
* **설정 정보**: 하이퍼파라미터, 데이터셋 링크 또는 사용 중인 아키텍처 이름을 구성 파라미터로 기록합니다. `wandb.init(config=your_config_dictionary)`와 같이 전달합니다. 자세한 내용은 [PyTorch Integrations]({{< relref path="/guides/integrations/pytorch.md" lang="ko" >}}) 페이지를 참조하십시오.
* **Metrics**: `wandb.log`를 사용하여 모델의 메트릭을 확인합니다. 트레이닝 루프 내에서 정확도 및 손실과 같은 메트릭을 기록하면 UI에서 라이브 업데이트 그래프를 얻을 수 있습니다.

## 일반적인 워크플로우

1. **최고 정확도 비교**: 여러 Runs에서 메트릭의 최고 값을 비교하려면 해당 메트릭에 대한 요약 값을 설정합니다. 기본적으로 요약은 각 키에 대해 기록한 마지막 값으로 설정됩니다. 이는 UI의 테이블에서 유용하며, 요약 메트릭을 기반으로 Runs를 정렬하고 필터링하여 최종 정확도 대신 _최고_ 정확도를 기준으로 테이블 또는 막대 차트에서 Runs를 비교하는 데 도움이 됩니다. 예를 들면 다음과 같습니다. `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **하나의 차트에 여러 메트릭**: 다음과 같이 `wandb.log`에 여러 메트릭을 한 번에 기록합니다. `wandb.log({"acc'": 0.9, "loss": 0.1})`. 그러면 UI에서 둘 다 플롯할 수 있습니다.
3. **사용자 정의 x축**: 동일한 로그 호출에 사용자 정의 x축을 추가하여 W&B 대시보드에서 다른 축을 기준으로 메트릭을 시각화합니다. 예를 들면 다음과 같습니다. `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`. 지정된 메트릭에 대한 기본 x축을 설정하려면 [Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ko" >}})을 사용하십시오.
4. **풍부한 미디어 및 차트 기록**: `wandb.log`는 [이미지 및 비디오와 같은 미디어]({{< relref path="./media.md" lang="ko" >}})에서 [테이블]({{< relref path="./log-tables.md" lang="ko" >}}) 및 [차트]({{< relref path="/guides/models/app/features/custom-charts/" lang="ko" >}})에 이르기까지 광범위한 데이터 유형의 로깅을 지원합니다.
