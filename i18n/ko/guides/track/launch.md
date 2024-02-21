---
description: Create a W&B Experiment.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 실험 생성하기

<head>
  <title>W&B 실험 시작하기</title>
</head>

W&B 파이썬 SDK를 사용하여 머신 러닝 실험을 추적하세요. 그런 다음 인터랙티브 대시보드에서 결과를 검토하거나 [W&B 공개 API](../../ref/python/public-api/README.md)를 통해 파이썬으로 데이터를 프로그래매틱하게 엑세스할 수 있습니다.

이 가이드는 W&B 구성 요소를 사용하여 W&B 실험을 생성하는 방법을 설명합니다.

## W&B 실험을 생성하는 방법

네 단계로 W&B 실험을 생성하세요:

1. [W&B 실행 초기화하기](#initialize-a-wb-run)
2. [하이퍼파라미터 딕셔너리 캡처하기](#capture-a-dictionary-of-hyperparameters)
3. [학습 루프 내에서 메트릭 로그하기](#log-metrics-inside-your-training-loop)
4. [W&B에 아티팩트 로그하기](#log-an-artifact-to-wb)

### W&B 실행 초기화하기
스크립트 호출의 시작 부분에서 [`wandb.init()`](../../ref/python/init.md) API를 사용하여 W&B 실행으로 동기화하고 데이터를 로그하는 백그라운드 프로세스를 생성하세요.

다음 코드 조각은 `“cat-classification”`이라는 새 W&B 프로젝트를 생성하는 방법을 보여줍니다. 이 실행을 식별하는 데 도움이 되는 메모 `“My first experiment”`가 추가되었습니다. `“baseline”` 및 `“paper1”` 태그는 이 실행이 미래의 논문 출판을 위한 기준 실험임을 상기시키기 위해 포함되었습니다.

```python
# W&B 파이썬 라이브러리 가져오기
import wandb

# 1. W&B 실행 시작하기
run = wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
)
```
`wandb.init()`으로 W&B를 초기화할 때 반환되는 [Run](../../ref/python/run.md) 객체입니다. 또한, W&B는 모든 로그와 파일이 저장되고 W&B 서버로 비동기적으로 스트리밍되는 로컬 디렉터리를 생성합니다.

:::info
참고: wandb.init()을 호출할 때 해당 프로젝트가 이미 존재하면 실행이 기존 프로젝트에 추가됩니다. 예를 들어, `“cat-classification”`이라는 프로젝트가 이미 있으면 그 프로젝트는 계속 존재하고 삭제되지 않습니다. 대신, 새 실행이 그 프로젝트에 추가됩니다.
:::

### 하이퍼파라미터 딕셔너리 캡처하기
학습률이나 모델 유형과 같은 하이퍼파라미터의 딕셔너리를 저장하세요. config에 캡처한 모델 설정은 나중에 결과를 정리하고 쿼리하는 데 유용합니다.

```python
#  2. 하이퍼파라미터 딕셔너리 캡처하기
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}
```
실험을 구성하는 방법에 대한 자세한 내용은 [실험 구성하기](./config.md)를 참조하세요.

### 학습 루프 내에서 메트릭 로그하기
각 `for` 루프(에포크) 동안 메트릭을 로그하세요, 정확도와 손실 값이 계산되어 [`wandb.log()`](../../ref/python/log.md)를 사용하여 W&B에 로그됩니다. wandb.log를 호출할 때 기본적으로 새 단계가 history 객체에 추가되고 summary 객체가 업데이트됩니다.

다음 코드 예제는 `wandb.log`를 사용하여 메트릭을 로그하는 방법을 보여줍니다.

:::note
모드를 설정하고 데이터를 검색하는 방법에 대한 세부 정보는 생략되었습니다.
:::

```python
# 모델 및 데이터 설정
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. 학습 루프 내에서 메트릭을 로그하여
        # 모델 성능을 시각화하세요
        wandb.log({"accuracy": accuracy, "loss": loss})
```
W&B로 로그할 수 있는 다양한 데이터 유형에 대한 자세한 내용은 [실험 중 데이터 로그하기](./log/intro.md)를 참조하세요.

### W&B에 아티팩트 로그하기
선택적으로 W&B 아티팩트를 로그하세요. 아티팩트는 데이터세트와 모델의 버전 관리를 쉽게 만듭니다.
```python
wandb.log_artifact(model)
```
아티팩트에 대한 자세한 내용은 [아티팩트 장](../artifacts/intro.md)을 참조하세요. 모델 버전 관리에 대한 자세한 내용은 [모델 관리](../model_registry/intro.md)를 참조하세요.

### 모든 것을 함께 두기
앞서 제시된 코드 조각으로 완성된 전체 스크립트는 아래에 있습니다:
```python
# W&B 파이썬 라이브러리 가져오기
import wandb

# 1. W&B 실행 시작하기
run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])

#  2. 하이퍼파라미터 딕셔너리 캡처하기
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}

# 모델 및 데이터 설정
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. 학습 루프 내에서 메트릭을 로그하여
        # 모델 성능을 시각화하세요
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. W&B에 아티팩트 로그하기
wandb.log_artifact(model)

# 선택적: 마지막에 모델 저장
model.to_onnx()
wandb.save("model.onnx")
```

## 다음 단계: 실험 시각화하기
W&B 대시보드를 사용하여 머신 러닝 모델의 결과를 조직하고 시각화하는 중앙 장소로 활용하세요. 몇 번의 클릭만으로 [평행 좌표 플롯](../app/features/panels/parallel-coordinates.md), [파라미터 중요도 분석](../app/features/panels/parameter-importance.md), 그리고 [더 많은](../app/features/panels/intro.md) 인터랙티브 차트를 구성할 수 있습니다.

![퀵스타트 스윕 대시보드 예시](/images/sweeps/quickstart_dashboard_example.png)

실험과 특정 실행을 보는 방법에 대한 자세한 내용은 [실험 결과 시각화하기](./app.md)를 참조하세요.

## 모범 사례
실험을 생성할 때 고려할 수 있는 몇 가지 제안된 지침은 다음과 같습니다:

1. **Config**: 하이퍼파라미터, 아키텍처, 데이터세트 등 모델을 재현하는 데 사용하고 싶은 것을 추적하세요. 이들은 열에 표시됩니다—앱에서 동적으로 실행을 그룹화, 정렬, 필터링하는 데 config 열을 사용하세요.
2. **프로젝트**: 프로젝트는 함께 비교할 수 있는 실험 세트입니다. 각 프로젝트는 전용 대시보드 페이지를 받으며, 다른 모델 버전을 비교하기 위해 다양한 실행 그룹을 쉽게 켜고 끌 수 있습니다.
3. **노트**: 자신에게 보내는 간단한 커밋 메시지입니다. 노트는 스크립트에서 설정할 수 있습니다. 나중에 W&B 앱의 프로젝트 대시보드 개요 섹션에서 노트를 편집할 수 있습니다.
4. **태그**: 기준 실행과 좋아하는 실행을 식별하세요. 태그를 사용하여 실행을 필터링할 수 있습니다. 나중에 W&B 앱의 프로젝트 대시보드 개요 섹션에서 태그를 편집할 수 있습니다.

다음 코드 조각은 위에 나열된 모범 사례를 사용하여 W&B 실험을 정의하는 방법을 보여줍니다:

```python
import wandb

config = dict(
    learning_rate=0.01, momentum=0.2, architecture="CNN", dataset_id="cats-0192"
)

wandb.init(
    project="detect-cats",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
)
```

W&B 실험을 정의할 때 사용할 수 있는 매개변수에 대한 자세한 내용은 [API 참조 가이드](../../ref/python/README.md)의 [`wandb.init`](../../ref/python/init.md) API 문서를 참조하세요.