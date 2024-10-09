---
title: Metrics and Performance FAQ
displayed_sidebar: default
---

## 메트릭

### 시스템 메트릭은 얼마나 자주 수집되나요?

기본적으로 메트릭은 매 2초마다 수집되며 15초 간격으로 평균화됩니다. 더 높은 해상도의 메트릭이 필요하면 [contact@wandb.com](mailto:contact@wandb.com)으로 이메일을 보내주세요.

### 코드나 데이터셋 예제 없이 메트릭만 로그할 수 있나요?

**Dataset 예제**

기본적으로, 우리는 당신의 데이터셋 예제를 로그하지 않습니다. 이 기능을 켜면 웹 인터페이스에서 예측값 예제를 볼 수 있습니다.

**Code Logging**

코드 로그를 끄는 두 가지 방법이 있습니다:

1. `WANDB_DISABLE_CODE`를 `true`로 설정하여 모든 코드 추적을 끕니다. 우리는 git SHA나 diff 패치를 수집하지 않습니다.
2. `WANDB_IGNORE_GLOBS`를 `*.patch`로 설정하여 diff 패치의 서버로의 동기화를 중지합니다. 로컬에는 그대로 남아 있어 `wandb restore`로 적용할 수 있습니다.

### 두 가지 다른 시간 간격으로 메트릭을 로그할 수 있나요? (예를 들어, 배치별 트레이닝 정확도와 에포크별 검증 정확도를 로그하고 싶습니다.)

네, 다른 메트릭을 로그할 때 지표(예: `batch`와 `epoch`)를 함께 로그하면 됩니다. 한 단계에서는 `wandb.log({'train_accuracy': 0.9, 'batch': 200})`를 호출하고, 다른 단계에서는 `wandb.log({'val_accuracy': 0.8, 'epoch': 4})`를 호출할 수 있습니다. 그러면 UI에서 각 차트에 대해 적절한 값을 x축으로 설정할 수 있습니다. 특정 지표의 기본 x축을 설정하려면 [Run.define_metric()](../../ref/python/run.md#define_metric)를 사용하여 설정할 수 있습니다. 위의 예에서 다음과 같이 할 수 있습니다:

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```

### 시간이 지나도 변하지 않는 최종 평가 정확도 같은 메트릭을 로그할 수 있나요?

`wandb.log({'final_accuracy': 0.9}`을 사용하면 됩니다. 기본적으로 `wandb.log({'final_accuracy'})`은 `wandb.settings['final_accuracy']`를 업데이트하며, 이는 runs 테이블에 표시되는 값입니다.

### run이 완료된 후 추가 메트릭을 로그할 수 있나요?

이를 위한 여러 방법이 있습니다.

복잡한 워크플로우의 경우, 여러 run을 사용하고 [wandb.init](../track/launch.md)의 그룹 파라미터를 고유한 값으로 설정하여 단일 실험의 일부로 실행되는 모든 프로세스에 적용하는 것을 권장합니다. [runs 테이블](../app/pages/run-page.md)은 자동으로 테이블을 그룹 ID별로 그룹화하고 시각화는 예상대로 동작합니다. 이를 통해 여러 Experiments 및 트레이닝 run을 개별 프로세스로 실행하고 모든 결과를 단일 장소로 로그할 수 있습니다.

더 간단한 워크플로우의 경우, `resume=True`와 `id=UNIQUE_ID`로 `wandb.init`을 호출한 다음 동일한 `id=UNIQUE_ID`로 다시 `wandb.init`을 호출할 수 있습니다. 그러면 [`wandb.log`](../track/log/intro.md) 또는 `wandb.summary`로 정상적으로 로그할 수 있으며 run 값들이 업데이트됩니다.

## 성능

### wandb가 트레이닝 속도를 저하시킬까요?

W&B는 일반적인 사용 시 트레이닝 성능에 거의 영향을 미치지 않아야 합니다. wandb의 일반적인 사용은 초당 한 번 이하로 로그하고 각 단계에서 몇 메가바이트 이하의 데이터를 로그하는 것입니다. W&B는 별도의 프로세스에서 실행되며 함수 호출이 차단되지 않으므로 네트워크가 일시적으로 끊기거나 디스크에서의 읽기/쓰기 문제가 발생하더라도 성능에 영향을 미치지 않습니다. 많은 양의 데이터를 빠르게 로그할 수 있으며, 그렇게 하면 디스크 I/O 문제가 발생할 수 있습니다. 질문이 있으면 언제든지 문의해 주세요.

### 프로젝트당 생성할 run의 수는 몇 개가 적절한가요?

성능 상의 이유로 프로젝트당 대략 10,000개의 run을 권장합니다.

### 하이퍼파라미터 검색을 조직하는 모범 사례

프로젝트당 10,000 run (대략)이 적절한 한계라면 `wandb.init()`에서 태그를 설정하고 각 검색에 대해 고유한 태그를 가지는 것을 추천합니다. 이렇게 하면 프로젝트 페이지의 Runs 테이블에서 해당 태그를 클릭하여 특정 검색으로 쉽게 필터링할 수 있습니다. 예를 들어 `wandb.init(tags='your_tag')`입니다. 이에 대한 문서는 [여기](../../ref/python/init.md)에서 찾을 수 있습니다.