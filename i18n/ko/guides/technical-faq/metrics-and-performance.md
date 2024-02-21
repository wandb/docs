---
displayed_sidebar: default
---

# 메트릭 & 성능

## 메트릭

### 시스템 메트릭이 얼마나 자주 수집되나요?

기본적으로 메트릭은 2초마다 수집되며 15초 동안 평균화됩니다. 더 높은 해상도의 메트릭이 필요하다면 [contact@wandb.com](mailto:contact@wandb.com)으로 이메일을 보내주세요.

### 코드나 데이터세트 예제 없이 메트릭만 로깅할 수 있나요?

**데이터세트 예제**

기본적으로 데이터세트 예제는 로깅하지 않습니다. 이 기능을 명시적으로 켜서 웹 인터페이스에서 예제 예측값을 볼 수 있습니다.

**코드 로깅**

코드 로깅을 끄는 두 가지 방법이 있습니다:

1. 모든 코드 추적을 끄려면 `WANDB_DISABLE_CODE`를 `true`로 설정하세요. git SHA나 diff 패치를 수집하지 않습니다.
2. diff 패치를 서버에 동기화하지 않으려면 `WANDB_IGNORE_GLOBS`를 `*.patch`로 설정하세요. 로컬에는 여전히 있으며 [wandb restore](../track/save-restore.md) 명령으로 적용할 수 있습니다.

### 두 가지 다른 시간 척도에서 메트릭을 로깅할 수 있나요? (예를 들어, 배치마다 학습 정확도를 로그하고 에포크마다 검증 정확도를 로그하고 싶습니다.)

네, 다른 메트릭을 로깅할 때마다 인덱스(예: `batch`와 `epoch`)를 로깅하면 됩니다. 한 단계에서 `wandb.log({'train_accuracy': 0.9, 'batch': 200})`를 호출하고 다른 단계에서 `wandb.log({'val_accuracy': 0.8, 'epoch': 4})`를 호출할 수 있습니다. 그런 다음 UI에서 각 차트의 x축으로 적절한 값을 설정할 수 있습니다. 특정 인덱스의 기본 x축을 설정하고 싶다면 [Run.define_metric()](../../ref/python/run.md#define_metric)를 사용하여 할 수 있습니다. 위의 예제에서는 다음과 같이 할 수 있습니다:

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```

### 시간이 지남에 따라 변하지 않는 메트릭(예: 최종 평가 정확도)을 어떻게 로깅할 수 있나요?

`wandb.log({'final_accuracy': 0.9}`를 사용하면 됩니다. 기본적으로 `wandb.log({'final_accuracy'})`는 `wandb.settings['final_accuracy']`를 업데이트하며, 이는 실행 테이블에 표시되는 값입니다.

### 실행이 완료된 후 추가 메트릭을 어떻게 로깅할 수 있나요?

여러 방법이 있습니다.

복잡한 워크플로의 경우, [`wandb.init`](../track/launch.md)에서 그룹 파라미터를 설정하고 단일 실험의 일부로 실행되는 모든 프로세스에서 고유한 값을 가지도록 권장합니다. [실행 테이블](../app/pages/run-page.md)은 자동으로 그룹 ID별로 테이블을 그룹화하고 시각화는 예상대로 동작합니다. 이를 통해 여러 실험과 학습 실행을 별도의 프로세스로 실행하고 모든 결과를 한 곳에 로깅할 수 있습니다.

보다 단순한 워크플로의 경우, `wandb.init`을 `resume=True` 및 `id=UNIQUE_ID`와 함께 호출한 다음 나중에 동일한 `id=UNIQUE_ID`를 사용하여 `wandb.init`을 다시 호출할 수 있습니다. 그런 다음 [`wandb.log`](../track/log/intro.md) 또는 `wandb.summary`로 정상적으로 로깅하면 실행 값이 업데이트됩니다.

## 성능

### wandb가 내 학습을 느리게 하나요?

wandb를 정상적으로 사용한다면 학습 성능에 미미한 영향을 미쳐야 합니다. wandb의 정상적인 사용은 초당 한 번 미만 로깅하고 각 단계에서 몇 메가바이트의 데이터를 로깅하는 것을 의미합니다. W&B는 별도의 프로세스에서 실행되며 함수 호출은 차단되지 않으므로 네트워크가 잠시 다운되거나 디스크의 간헐적인 읽기/쓰기 문제가 있어도 성능에 영향을 미치지 않아야 합니다. 대량의 데이터를 빠르게 로깅하면 디스크 I/O 문제가 발생할 수 있습니다. 궁금한 점이 있으면 주저하지 말고 연락주세요.

### 프로젝트 당 생성할 실행 수는?

성능상의 이유로 프로젝트 당 대략 1만 개의 실행을 권장합니다.

### 하이퍼파라미터 탐색을 구성하는 모범 사례

프로젝트 당 약 1만 개의 실행(대략)이 합리적인 제한이라면, `wandb.init()`에서 태그를 설정하고 각 탐색에 대해 고유한 태그를 가지는 것이 좋습니다. 이렇게 하면 프로젝트 페이지의 실행 테이블에서 해당 태그를 클릭하여 프로젝트를 특정 탐색으로 쉽게 필터링할 수 있습니다. 예를 들어 `wandb.init(tags='your_tag')` 문서는 [여기](../../ref/python/init.md)에서 찾을 수 있습니다.