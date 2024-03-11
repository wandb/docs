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

W&B Python SDK를 사용하여 기계학습 실험을 추적하세요. 그러면 인터랙티브 대시보드에서 결과를 검토하거나 [W&B Public API](../../ref/python/public-api/README.md)를 사용하여 Python으로 데이터를 프로그래매틱하게 엑세스할 수 있습니다.

이 가이드에서는 W&B 빌딩 블록을 사용하여 W&B 실험을 생성하는 방법을 설명합니다.

## W&B 실험 생성 방법

네 단계로 W&B 실험을 생성하세요:

1. [W&B Run 초기화하기](#initialize-a-wb-run)
2. [하이퍼파라미터의 사전 캡처하기](#capture-a-dictionary-of-hyperparameters)
3. [트레이닝 루프 내에서 메트릭 로그하기](#log-metrics-inside-your-training-loop)
4. [W&B에 아티팩트 로그하기](#log-an-artifact-to-wb)

### W&B Run 초기화하기
스크립트 시작 부분에서 [`wandb.init()`](../../ref/python/init.md) API를 호출하여 W&B Run으로 데이터를 동기화하고 로그하는 백그라운드 프로세스를 생성하세요.

다음 코드 조각은 `“cat-classification”`이라는 새 W&B 프로젝트를 생성하는 방법을 보여줍니다. `“My first experiment”`라는 메모가 이 run을 식별하는 데 도움이 되도록 추가되었습니다. 이 run이 추후 논문 출판을 위한 베이스라인 실험이라는 것을 상기시키기 위해 `“baseline”`, `“paper1”` 태그가 포함되어 있습니다.

```python
# W&B Python 라이브러리 임포트
import wandb

# 1. W&B Run 시작하기
run = wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
)
```
`wandb.init()`으로 W&B를 초기화할 때 [Run](../../ref/python/run.md) 오브젝트가 반환됩니다. 추가로, W&B는 모든 로그와 파일이 저장되고 W&B 서버로 비동기적으로 스트리밍되는 로컬 디렉토리를 생성합니다.

:::info
안내: wandb.init()을 호출할 때 이미 해당 프로젝트가 존재하면, 실행은 기존 프로젝트에 추가됩니다. 예를 들어, 이미 `“cat-classification”`이라는 프로젝트가 있다면, 그 프로젝트는 계속 존재하며 삭제되지 않습니다. 대신, 그 프로젝트에 새 run이 추가됩니다.
:::

### 하이퍼파라미터의 사전 캡처하기
학습률이나 모델 유형과 같은 하이퍼파라미터의 사전을 저장하세요. config에 캡처한 모델 설정은 나중에 결과를 조직하고 쿼리할 때 유용합니다.

```python
#  2. 하이퍼파라미터의 사전 캡처하기
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}
```
실험을 구성하는 방법에 대한 자세한 정보는 [실험 구성하기](./config.md)를 참조하세요.

### 트레이닝 루프 내에서 메트릭 로그하기
각 `for` 루프(에포크) 동안 메트릭을 로그하면, 정확도와 손실 값이 계산되고 [`wandb.log()`](../../ref/python/log.md)를 사용하여 W&B에 로그됩니다. 기본적으로, wandb.log를 호출하면 새로운 단계가 history 오브젝트에 추가되고 summary 오브젝트가 업데이트됩니다.

다음 코드 예제는 `wandb.log`를 사용하여 메트릭을 로그하는 방법을 보여줍니다.

:::note
모델을 설정하고 데이터를 검색하는 방법의 세부 사항은 생략됩니다.
:::

```python
# 모델과 데이터 설정
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. 트레이닝 루프 내에서 메트릭 로그하기로 모델 성능 시각화
        wandb.log({"accuracy": accuracy, "loss": loss})
```
W&B로 로그할 수 있는 다양한 데이터 유형에 대한 자세한 정보는 [실험 중 데이터 로그하기](./log/intro.md)를 참조하세요.

### W&B에 아티팩트 로그하기
선택적으로 W&B 아티팩트를 로그하세요. 아티팩트는 데이터셋과 모델의 버전 관리를 쉽게 만듭니다.
```python
wandb.log_artifact(model)
```
아티팩트에 대한 자세한 정보는 [아티팩트 챕터](../artifacts/intro.md)를 참조하세요. 모델 버전 관리에 대한 자세한 정보는 [모델 관리](../model_registry/intro.md)를 참조하세요.

### 모든 것을 함께 놓기
앞서 언급된 코드 조각으로 구성된 전체 스크립트는 아래에 있습니다:
```python
# W&B Python 라이브러리 임포트
import wandb

# 1. W&B Run 시작하기
run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])

#  2. 하이퍼파라미터의 사전 캡처하기
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}

# 모델과 데이터 설정
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. 트레이닝 루프 내에서 메트릭 로그하기로 모델 성능 시각화
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. W&B에 아티팩트 로그하기
wandb.log_artifact(model)

# 선택사항: 마지막에 모델 저장
model.to_onnx()
wandb.save("model.onnx")
```

## 다음 단계: 실험 시각화하기
W&B 대시보드를 기계학습 모델의 결과를 조직하고 시각화하기 위한 중앙 장소로 사용하세요. 몇 번의 클릭만으로 [평행 좌표 플롯](../app/features/panels/parallel-coordinates.md), [파라미터 중요도 분석](../app/features/panels/parameter-importance.md), 그리고 [더 많은 것들](../app/features/panels/intro.md)과 같은 풍부하고 인터랙티브한 차트를 구성할 수 있습니다.

![퀵스타트 스윕 대시보드 예시](/images/sweeps/quickstart_dashboard_example.png)

실험과 특정 run을 보는 방법에 대한 자세한 정보는 [실험 결과 시각화하기](./app.md)를 참조하세요.

## 모범 사례
실험을 생성할 때 고려할 수 있는 몇 가지 제안된 지침입니다:

1. **Config**: 하이퍼파라미터, 아키텍처, 데이터셋 등 모델을 재현하기 위해 사용하고 싶은 모든 것을 추적하세요. 이들은 열에 표시됩니다—앱에서 동적으로 실행을 그룹화, 정렬, 필터링하기 위해 config 열을 사용하세요.
2. **프로젝트**: 프로젝트는 함께 비교할 수 있는 실험 세트입니다. 각 프로젝트는 전용 대시보드 페이지를 받으며, 다른 모델 버전을 비교하기 위해 다양한 그룹의 실행을 쉽게 켜고 끌 수 있습니다.
3. **노트**: 자신에게 보내는 간단한 커밋 메시지입니다. 노트는 스크립트에서 설정할 수 있습니다. 나중에 W&B 앱의 프로젝트 대시보드의 개요 섹션에서 노트를 편집할 수 있습니다.
4. **태그**: 베이스라인 실행과 좋아하는 실행을 식별하세요. 태그를 사용하여 실행을 필터링할 수 있습니다. 나중에 W&B 앱의 프로젝트 대시보드의 개요 섹션에서 태그를 편집할 수 있습니다.

다음 코드 조각은 위에 나열된 모범 사례를 사용하여 W&B 실험을 정의하는 방법을 보여줍니다:

```python
import wandb

config = dict(
    learning_rate=0.01, momentum=0.2, 아키텍처="CNN", dataset_id="cats-0192"
)

wandb.init(
    project="detect-cats",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
)
```

W&B 실험을 정의할 때 사용할 수 있는 매개변수에 대한 자세한 정보는 [`wandb.init`](../../ref/python/init.md) API 문서를 [API 참조 가이드](../../ref/python/README.md)에서 확인하세요.