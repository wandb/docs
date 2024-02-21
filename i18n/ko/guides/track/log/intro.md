---
description: Keep track of metrics, videos, custom plots, and more
slug: /guides/track/log
displayed_sidebar: default
---

# 실험에서 미디어와 개체 로깅하기

<head>
  <title>실험에서 미디어와 개체 로깅하기</title>
</head>

W&B Python SDK를 사용하여 단계별로 메트릭, 미디어 또는 사용자 정의 개체의 사전을 로그하세요. W&B는 각 단계에서 키-값 쌍을 수집하고 `wandb.log()`로 데이터를 로그할 때마다 통합된 하나의 사전에 저장합니다. 스크립트에서 로그된 데이터는 로컬 머신의 `wandb`라는 디렉터리에 저장된 후 W&B 클라우드 또는 [개인 서버](../../hosting/intro.md)로 동기화됩니다.

:::info
키-값 쌍은 각 단계에 대해 동일한 값을 전달하는 경우에만 하나의 통합 사전에 저장됩니다. `step`에 대해 다른 값을 로그하면 W&B는 수집된 모든 키와 값을 메모리에 씁니다.
:::

기본적으로 `wandb.log` 호출은 새로운 `step`입니다. W&B는 차트와 패널을 만들 때 기본 x축으로 단계를 사용합니다. 선택적으로 사용자 정의 x축을 생성하거나 사용자 정의 요약 메트릭을 캡처할 수 있습니다. 자세한 정보는 [로그 축 사용자 정의](./customize-logging-axes.md)를 참조하세요.




:::caution
각 `step`에 대해 연속적인 값을 로그하려면 `wandb.log()`를 사용하세요: 0, 1, 2 등. 특정 기록 단계에 쓰는 것은 불가능합니다. W&B는 "현재"와 "다음" 단계에만 씁니다.
:::

## 자동으로 로그되는 데이터

W&B 실험 중에 W&B는 다음 정보를 자동으로 로그합니다:

* **시스템 메트릭**: CPU 및 GPU 사용량, 네트워크 등. 이들은 [실행 페이지](../../app/pages/run-page.md)의 시스템 탭에 표시됩니다. GPU의 경우, [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface)로 가져옵니다.
* **명령 줄**: stdout과 stderr이 가져와져서 [실행 페이지](../../app/pages/run-page.md)의 로그 탭에 표시됩니다.

계정의 [설정 페이지](https://wandb.ai/settings)에서 [코드 저장](http://wandb.me/code-save-colab)을 켜서 로그하세요:

* **Git 커밋**: 최신 git 커밋을 가져와 실행 페이지의 overview 탭에 표시되며, 커밋되지 않은 변경 사항이 있을 경우 `diff.patch` 파일도 보입니다.
* **의존성**: `requirements.txt` 파일이 업로드되어 실행 페이지의 파일 탭에 표시되며, 실행을 위해 `wandb` 디렉터리에 저장한 모든 파일과 함께 표시됩니다.

## 특정 W&B API 호출로 로그되는 데이터는 무엇인가요?

W&B를 사용하면 로그하고 싶은 것을 정확히 결정할 수 있습니다. 다음은 일반적으로 로그되는 개체 목록입니다:

* **데이터세트**: 이미지나 다른 데이터세트 샘플을 특별히 로그해야 W&B로 스트리밍됩니다.
* **플롯**: `wandb.plot`을 `wandb.log`와 함께 사용하여 차트를 추적하세요. 자세한 정보는 [로그 플롯](./plots.md)을 참조하세요.
* **테이블**: `wandb.Table`을 사용하여 W&B와 함께 시각화하고 쿼리할 데이터를 로그하세요. 자세한 정보는 [로그 테이블](./log-tables.md)을 참조하세요.
* **PyTorch 그레이디언트**: UI에서 가중치의 그레이디언트를 히스토그램으로 보려면 `wandb.watch(model)`을 추가하세요.
* **구성 정보**: 하이퍼파라미터, 데이터세트 링크 또는 사용 중인 아키텍처 이름과 같은 구성 파라미터를 다음과 같이 로그하세요: `wandb.init(config=your_config_dictionary)`. 자세한 정보는 [PyTorch 통합](../../integrations/pytorch.md) 페이지를 참조하세요.
* **메트릭**: `wandb.log`를 사용하여 모델의 메트릭을 확인하세요. 학습 루프 내부에서 정확도와 손실과 같은 메트릭을 로그하면 UI에서 실시간으로 업데이트되는 그래프를 얻을 수 있습니다.

## 일반적인 워크플로

1. **최고 정확도 비교**: 메트릭의 최고 값이 실행 간에 어떻게 비교되는지 확인하려면 해당 메트릭에 대한 요약 값을 설정하세요. 기본적으로 요약은 각 키에 대해 로그한 마지막 값을 설정합니다. 이는 UI의 테이블에서 실행을 요약 메트릭을 기준으로 정렬하고 필터링할 수 있으므로 유용합니다. 즉, 최종 정확도가 아닌 _최고_ 정확도를 기준으로 테이블이나 막대 차트에서 실행을 비교할 수 있습니다. 예를 들어 요약을 다음과 같이 설정할 수 있습니다: `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **하나의 차트에서 여러 메트릭**: `wandb.log({"acc'": 0.9, "loss": 0.1})`와 같이 같은 호출로 여러 메트릭을 로그하면 UI에서 플롯할 수 있습니다.
3. **사용자 정의 x축**: 다른 축에서 메트릭을 시각화하려면 동일한 로그 호출에 사용자 정의 x축을 추가하세요. 예를 들어: `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`. 주어진 메트릭에 대한 기본 x축을 설정하려면 [Run.define\_metric()](../../../ref/python/run.md#define_metric)를 사용하세요.
4. **풍부한 미디어와 차트 로깅**: `wandb.log`는 [이미지와 비디오](./media.md)와 같은 미디어부터 [테이블](./log-tables.md)과 [차트](../../app/features/custom-charts/intro.md)와 같은 다양한 데이터 유형을 로깅하는 것을 지원합니다.