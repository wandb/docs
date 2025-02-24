---
title: Create an experiment
description: W&B 실험 을 만듭니다.
menu:
  default:
    identifier: ko-guides-models-track-launch
    parent: experiments
weight: 1
---

W&B Python SDK를 사용하여 기계 학습 Experiments를 추적하세요. 그런 다음 인터랙티브 대시보드에서 결과를 검토하거나 [W&B Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})로 데이터에 프로그래밍 방식으로 엑세스하기 위해 Python으로 데이터를 내보낼 수 있습니다.

이 가이드에서는 W&B 구성 요소를 사용하여 W&B Experiment를 만드는 방법을 설명합니다.

## W&B Experiment를 만드는 방법

다음 네 단계로 W&B Experiment를 만듭니다.

1. [W&B Run 초기화]({{< relref path="#initialize-a-wb-run" lang="ko" >}})
2. [하이퍼파라미터 사전 캡처]({{< relref path="#capture-a-dictionary-of-hyperparameters" lang="ko" >}})
3. [트레이닝 루프 내에서 메트릭 기록]({{< relref path="#log-metrics-inside-your-training-loop" lang="ko" >}})
4. [W&B에 아티팩트 기록]({{< relref path="#log-an-artifact-to-wb" lang="ko" >}})

### W&B run 초기화
스크립트 호출의 시작 부분에서 [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}) API를 호출하여 W&B Run으로 데이터를 동기화하고 기록하는 백그라운드 프로세스를 생성합니다.

다음 코드조각은 `“cat-classification”`이라는 새 W&B project를 만드는 방법을 보여줍니다. 이 run을 식별하는 데 도움이 되도록 `“My first experiment”`라는 메모가 추가되었습니다. 태그 `“baseline”`과 `“paper1”`은 이 run이 향후 논문 출판을 위한 베이스라인 experiment임을 상기시켜 줍니다.

```python
# W&B Python Library 가져오기
import wandb

# 1. W&B Run 시작
run = wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
)
```
`wandb.init()`으로 W&B를 초기화하면 [Run]({{< relref path="/ref/python/run.md" lang="ko" >}}) 오브젝트가 반환됩니다. 또한 W&B는 모든 로그와 파일이 저장되고 W&B server로 비동기적으로 스트리밍되는 로컬 디렉토리를 만듭니다.

{{% alert %}}
참고: wandb.init()을 호출할 때 해당 project가 이미 존재하는 경우 Runs가 기존 project에 추가됩니다. 예를 들어 `“cat-classification”`이라는 project가 이미 있는 경우 해당 project는 계속 존재하며 삭제되지 않습니다. 대신 새 run이 해당 project에 추가됩니다.
{{% /alert %}}

### 하이퍼파라미터 사전 캡처
학습률 또는 모델 유형과 같은 하이퍼파라미터 사전을 저장합니다. config에 캡처하는 모델 설정은 나중에 결과를 구성하고 쿼리하는 데 유용합니다.

```python
#  2. 하이퍼파라미터 사전 캡처
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}
```
experiment를 구성하는 방법에 대한 자세한 내용은 [Experiments 구성]({{< relref path="./config.md" lang="ko" >}})을 참조하세요.

### 트레이닝 루프 내에서 메트릭 기록
각 `for` 루프(에포크) 동안 메트릭을 기록합니다. 정확도와 손실 값이 계산되어 [`wandb.log()`]({{< relref path="/ref/python/log.md" lang="ko" >}})로 W&B에 기록됩니다. 기본적으로 wandb.log를 호출하면 기록 오브젝트에 새 단계가 추가되고 요약 오브젝트가 업데이트됩니다.

다음 코드 예제는 `wandb.log`로 메트릭을 기록하는 방법을 보여줍니다.

{{% alert %}}
모드를 설정하고 데이터를 검색하는 방법에 대한 세부 정보는 생략되었습니다.
{{% /alert %}}

```python
# 모델 및 데이터 설정
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. 트레이닝 루프 내에서 메트릭을 기록하여
        # 모델 성능을 시각화합니다.
        wandb.log({"accuracy": accuracy, "loss": loss})
```
W&B로 기록할 수 있는 다양한 데이터 유형에 대한 자세한 내용은 [Experiments 중 데이터 기록]({{< relref path="/guides/models/track/log/" lang="ko" >}})을 참조하세요.

### W&B에 아티팩트 기록
선택적으로 W&B Artifact를 기록합니다. Artifact를 사용하면 데이터셋과 Models를 쉽게 버전 관리할 수 있습니다.
```python
wandb.log_artifact(model)
```
Artifacts에 대한 자세한 내용은 [Artifacts 챕터]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 참조하세요. 모델 버전 관리에 대한 자세한 내용은 [모델 관리]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})를 참조하세요.

### 모두 함께 놓기
위의 코드조각이 포함된 전체 스크립트는 다음과 같습니다.
```python
# W&B Python Library 가져오기
import wandb

# 1. W&B Run 시작
run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])

#  2. 하이퍼파라미터 사전 캡처
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}

# 모델 및 데이터 설정
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. 트레이닝 루프 내에서 메트릭을 기록하여
        # 모델 성능을 시각화합니다.
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. W&B에 아티팩트 기록
wandb.log_artifact(model)

# 선택 사항: 마지막에 모델 저장
model.to_onnx()
wandb.save("model.onnx")
```

## 다음 단계: experiment 시각화
W&B Dashboard를 기계 학습 모델의 결과를 구성하고 시각화하는 중앙 위치로 사용하세요. 몇 번의 클릭만으로 [평행 좌표 플롯]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ko" >}}), [파라미터 중요도 분석]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ko" >}}) 및 [기타]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}})와 같은 풍부한 인터랙티브 차트를 구성하세요.

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Quickstart Sweeps Dashboard example" >}}

Experiments 및 특정 runs를 보는 방법에 대한 자세한 내용은 [Experiments 결과 시각화]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})를 참조하세요.

## 모범 사례
다음은 Experiments를 만들 때 고려해야 할 몇 가지 제안된 지침입니다.

1. **Config**: 하이퍼파라미터, 아키텍처, 데이터셋 및 모델을 재현하는 데 사용하려는 모든 항목을 추적합니다. 이는 열에 표시됩니다. config 열을 사용하여 앱에서 runs를 동적으로 그룹화, 정렬 및 필터링합니다.
2. **Project**: project는 함께 비교할 수 있는 Experiments 집합입니다. 각 project에는 전용 대시보드 페이지가 있으며, 다양한 모델 버전을 비교하기 위해 다양한 runs 그룹을 쉽게 켜고 끌 수 있습니다.
3. **Notes**: 스크립트에서 직접 빠른 커밋 메시지를 설정합니다. W&B App에서 run의 개요 섹션에서 메모를 편집하고 엑세스합니다.
4. **Tags**: 베이스라인 runs와 즐겨찾는 runs를 식별합니다. 태그를 사용하여 runs를 필터링할 수 있습니다. W&B App의 project 대시보드 개요 섹션에서 나중에 태그를 편집할 수 있습니다.

다음 코드조각은 위에 나열된 모범 사례를 사용하여 W&B Experiment를 정의하는 방법을 보여줍니다.

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

W&B Experiment를 정의할 때 사용할 수 있는 파라미터에 대한 자세한 내용은 [API Reference Guide]({{< relref path="/ref/python/" lang="ko" >}})의 [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ko" >}}) API 문서를 참조하세요.
