---
title: Create an experiment
description: W&B Experiment 생성.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Python SDK를 사용하여 기계학습 실험을 추적하세요. 그런 다음 결과를 인터랙티브 대시보드에서 검토하거나, 데이터를 Python으로 내보내서 [W&B 공용 API](../../ref/python/public-api/README.md)를 통해 프로그래밍적으로 엑세스할 수 있습니다.

이 가이드는 W&B 빌딩 블록을 사용하여 W&B Experiment를 생성하는 방법을 설명합니다.

## W&B Experiment 만드는 방법

W&B Experiment는 다음 네 단계로 생성할 수 있습니다:

1. [W&B Run 초기화하기](#initialize-a-wb-run)
2. [하이퍼파라미터 사전 캡처하기](#capture-a-dictionary-of-hyperparameters)
3. [트레이닝 루프 내에서 메트릭 로그하기](#log-metrics-inside-your-training-loop)
4. [W&B에 아티팩트 로그하기](#log-an-artifact-to-wb)

### W&B run 초기화하기
스크립트 호출 시작 시, [`wandb.init()`](../../ref/python/init.md) API를 사용하여 W&B Run으로 데이터를 동기화하고 로그하기 위한 백그라운드 프로세스를 생성합니다.

다음 코드조각은 `“cat-classification”`이라는 이름의 새로운 W&B 프로젝트를 생성하는 방법을 보여줍니다. 이 run을 식별하는 데 도움을 주기 위해 `“My first experiment”`라는 노트가 추가되었습니다. 이 run이 미래 논문 발표를 위한 베이스라인 실험임을 상기시키기 위해 태그 `“baseline”`과 `“paper1”`이 포함되어 있습니다.

```python
# Import the W&B Python Library
import wandb

# 1. Start a W&B Run
run = wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
)
```
`wandb.init()`로 W&B를 초기화할 때 [Run](../../ref/python/run.md) 오브젝트가 반환됩니다. 또한 W&B는 모든 로그와 파일이 저장되고 비동기적으로 W&B 서버로 스트리밍되는 로컬 디렉토리를 생성합니다.

:::info
참고: run은 wandb.init()를 호출할 때 이미 존재하는 프로젝트에 추가됩니다. 예를 들어 `“cat-classification”`이라는 프로젝트가 이미 있을 경우, 해당 프로젝트는 계속 존재하고 삭제되지 않습니다. 대신 새 run이 해당 프로젝트에 추가됩니다.
:::

### 하이퍼파라미터 사전 캡처하기
학습률 또는 모델 타입과 같은 하이퍼파라미터의 사전을 저장하세요. config에서 캡처한 모델 설정은 나중에 결과를 조직하고 쿼리하는 데 유용합니다.

```python
#  2. Capture a dictionary of hyperparameters
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}
```
실험을 구성하는 방법에 대한 자세한 내용은 [Configure Experiments](./config.md)를 참조하세요.

### 트레이닝 루프 내에서 메트릭 로그하기
각 `for` 루프(에포크) 동안 메트릭을 로그합니다. 정확도와 손실 값은 계산되어 [`wandb.log()`](../../ref/python/log.md)를 사용하여 W&B에 로그됩니다. 기본적으로 wandb.log를 호출하면 히스토리 오브젝트의 새 스텝이 추가되고 요약 오브젝트가 업데이트됩니다.

다음 코드 예제는 `wandb.log`를 사용하여 메트릭을 로그하는 방법을 보여줍니다.

:::note
모델 설정 및 데이터 검색 방법에 대한 세부 정보는 생략되었습니다.
:::

```python
# Set up model and data
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. Log metrics inside your training loop to visualize
        # model performance
        wandb.log({"accuracy": accuracy, "loss": loss})
```
W&B로 로그할 수 있는 다양한 데이터 유형에 대한 자세한 내용은 [Log Data During Experiments](./log/intro.md)를 참조하세요.

### W&B에 아티팩트 로그하기
선택적으로 W&B Artifact를 로그하세요. Artifacts를 사용하면 데이터셋과 모델을 쉽게 버전 관리할 수 있습니다.
```python
wandb.log_artifact(model)
```
Artifacts에 대한 자세한 내용은 [Artifacts Chapter](../artifacts/intro.md)를 참조하세요. 모델 버전 관리에 대한 자세한 내용은 [Model Management](../model_registry/intro.md)를 참조하세요.

### 모두 모아 보기
이전 코드조각과 함께 전체 스크립트는 다음과 같습니다:
```python
# Import the W&B Python Library
import wandb

# 1. Start a W&B Run
run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])

#  2. Capture a dictionary of hyperparameters
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}

# Set up model and data
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. Log metrics inside your training loop to visualize
        # model performance
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. Log an artifact to W&B
wandb.log_artifact(model)

# Optional: save model at the end
model.to_onnx()
wandb.save("model.onnx")
```

## 다음 단계: 실험 시각화
W&B 대시보드를 사용하여 기계학습 모델에서 얻은 결과를 조직하고 시각화하는 중앙 장소로 활용하세요. 몇 번의 클릭만으로 [평행 좌표 플롯](../app/features/panels/parallel-coordinates.md), [파라미터 중요도 분석](../app/features/panels/parameter-importance.md) 등의 풍부한 인터랙티브 차트를 구성할 수 있습니다.

![퀵스타트 스윕 대시보드 예제](/images/sweeps/quickstart_dashboard_example.png)

실험과 특정 run을 시각화하는 방법에 대한 자세한 내용은 [Visualize results from experiments](./app.md)를 참조하세요.

## 모범 사례
다음은 실험을 생성할 때 고려해야 할 몇 가지 권장 지침입니다:

1. **Config**: 하이퍼파라미터, 아키텍처, 데이터셋 및 모델 재현에 사용하고 싶은 모든 것을 추적하세요. 이러한 항목은 열에 표시됩니다. app에서 config 열을 사용하여 run을 동적으로 그룹화, 정렬 및 필터링하세요.
2. **Project**: 프로젝트는 함께 비교할 수 있는 실험 모음입니다. 각 프로젝트는 전용 대시보드 페이지를 가지며, 다른 모델 버전을 비교하기 위해 run 그룹을 쉽게 켜고 끌 수 있습니다.
3. **Notes**: 스스로에게 빠른 커밋 메시지를 남기세요. 노트는 스크립트에서 설정할 수 있습니다. 나중에 W&B App의 프로젝트 대시보드 개요 섹션에서 노트를 편집할 수 있습니다.
4. **Tags**: 베이스라인 run과 좋아하는 run을 식별하세요. 태그를 사용하여 run을 필터링할 수 있습니다. 나중에 W&B App의 프로젝트 대시보드 개요 섹션에서 태그를 편집할 수 있습니다.

다음 코드조각은 위에 나열된 모범 사례를 사용하여 W&B Experiment를 정의하는 방법을 보여줍니다:

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

W&B Experiment를 정의할 때 사용할 수 있는 파라미터에 대한 자세한 내용은 [`wandb.init`](../../ref/python/init.md) API 문서를 [API Reference Guide](../../ref/python/README.md)에서 참조하세요.