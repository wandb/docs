---
title: 실험 만들기
description: W&B Experiment 만들기
menu:
  default:
    identifier: ko-create-an-experiment
    parent: experiments
weight: 1
---

W&B Python SDK를 사용해 기계학습 experiment 를 추적하세요. 이후 결과를 인터랙티브 대시보드에서 확인하거나, [W&B Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 통해 Python으로 데이터를 내보내 활용할 수 있습니다.

이 가이드에서는 W&B의 빌딩 블록들을 활용하여 W&B Experiment 를 만드는 방법을 설명합니다.

## W&B Experiment 생성 방법

W&B Experiment 는 네 단계로 생성합니다:

1. [W&B Run 초기화]({{< relref path="#initialize-a-wb-run" lang="ko" >}})
2. [하이퍼파라미터 딕셔너리 캡처]({{< relref path="#capture-a-dictionary-of-hyperparameters" lang="ko" >}})
3. [트레이닝 루프 안에서 메트릭 로깅]({{< relref path="#log-metrics-inside-your-training-loop" lang="ko" >}})
4. [W&B에 artifact 로깅]({{< relref path="#log-an-artifact-to-wb" lang="ko" >}})

### W&B run 초기화하기
[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})를 사용해 W&B Run 을 시작할 수 있습니다.

아래 코드 예제는 `"cat-classification"`이라는 W&B Project 에서 run 을 생성하는 방법을 보여줍니다. `“My first experiment”`라는 설명을 추가해 run 을 쉽게 구분하고, `“baseline”`, `“paper1”` 태그를 달아 해당 run 이 논문 출판을 위한 베이스라인임을 나타냅니다.

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
) as run:
    ...
```

`wandb.init()`는 [Run]({{< relref path="/ref/python/sdk/classes/run" lang="ko" >}}) 오브젝트를 반환합니다.

{{% alert %}}
안내: `wandb.init()`를 호출할 때 이미 존재하는 Project 가 있으면, Run 이 해당 Project 에 추가됩니다. 예를 들어 `"cat-classification"` Project 가 이미 존재한다면, 기존 Project 는 그대로 두고 새 run 만 추가됩니다.
{{% /alert %}}

### 하이퍼파라미터 딕셔너리 캡처
러닝레이트, 모델 타입 등 하이퍼파라미터 정보를 딕셔너리 형태로 저장하세요. config 에 저장한 설정 값들은 나중에 결과를 정리하거나 쿼리할 때 유용하게 활용됩니다.

```python
with wandb.init(
    ...,
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    ...
```

실험 설정과 관련된 자세한 내용은 [Experiments 설정하기]({{< relref path="./config.md" lang="ko" >}})를 참고하세요.

### 트레이닝 루프 안에서 메트릭 로깅
[`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ko" >}})를 통해 에포크마다 accuracy, loss 같은 메트릭을 기록하세요.

```python
model, dataloader = get_model(), get_data()

for epoch in range(run.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        run.log({"accuracy": accuracy, "loss": loss})
```

W&B로 기록할 수 있는 다양한 데이터 타입에 대해 궁금하다면 [트레이닝 중 데이터 로깅하기]({{< relref path="/guides/models/track/log/" lang="ko" >}})를 참고하세요.

### W&B에 artifact 로깅하기
원한다면 W&B Artifacts 를 기록할 수도 있습니다. Artifacts 는 데이터셋과 모델을 버전 관리할 수 있도록 도와줍니다.
```python
# 파일이나 디렉토리도 저장할 수 있습니다.
# 여기서는 모델 객체가 save() 메서드를 갖고 있다고 가정하고,
# ONNX 파일로 저장하는 예시입니다.
model.save("path_to_model.onnx")
run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```
[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에 대한 더 자세한 안내나, [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})를 활용한 모델 버전 관리 방법을 참고하세요.

### 전체 코드 결합하기
위에서 설명한 모든 요소를 하나로 합친 전체 스크립트 예시입니다:

```python
import wandb

with wandb.init(
    project="cat-classification",
    notes="",
    tags=["baseline", "paper1"],
    # Run의 하이퍼파라미터 기록
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    # 모델과 데이터셋 준비
    model, dataloader = get_model(), get_data()

    # 메트릭 로깅하면서 트레이닝 수행
    for epoch in range(run.config["epochs"]):
        for batch in dataloader:
            loss, accuracy = model.training_step()
            run.log({"accuracy": accuracy, "loss": loss})

    # 학습된 모델을 artifact로 업로드
    model.save("path_to_model.onnx")
    run.log_artifact("path_to_model.onnx", name="trained-model", type="model")
```

## 다음 단계: experiment 시각화 
W&B Dashboard 는 기계학습 모델 결과를 정리하고 시각화하는 중심지입니다. 클릭 몇 번만으로 [평행좌표 플롯]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ko" >}}), [파라미터 중요도]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ko" >}}) 등 다양한 차트 유형을 풍부하고 인터랙티브하게 만들 수 있습니다.

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Quickstart Sweeps Dashboard example" >}}

experiment 와 개별 run 결과를 확인하는 자세한 방법은 [실험 결과 시각화]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})를 참조하세요.

## 모범 사례
experiment 관리에 도움이 될 몇 가지 권장 가이드라인을 소개합니다:

1. **Run 종료**: `wandb.init()`를 `with` 구문 내부에서 사용하면, 코드가 끝나거나 예외가 발생할 때 run 이 자동으로 종료(완료)됩니다.
    * Jupyter 노트북에서는 Run 오브젝트를 직접 관리하는 편이 더 편할 수 있습니다. 이 경우에는 Run 에서 `finish()`를 직접 호출하세요:

        ```python
        # 노트북 셀에서:
        run = wandb.init()

        # 다른 셀에서:
        run.finish()
        ```
2. **Config**: 하이퍼파라미터, 아키텍처, 데이터셋 등 모델 재현에 필요한 정보를 빠짐 없이 config에 기록하세요. 이 값들은 앱의 컬럼에 표시되며, 그룹화/정렬/필터링에 유용합니다.
3. **Project**: Project 는 비교가 가능한 experiment 들의 모음입니다. 각각의 Project 마다 대시보드 페이지가 자동으로 생성되어 여러 run 그룹을 쉽게 관리하고, 다양한 모델 버전을 비교할 수 있습니다.
4. **Notes**: 커밋 메시지처럼, 스크립트에서 바로 간단한 설명을 남길 수 있습니다. 남긴 노트는 W&B App의 Run Overview 섹션에서 열람·수정이 가능합니다.
5. **Tags**: 베이스라인 run 이나 중요한 run 에 태그를 달아 구분하세요. 태그별로 run 을 필터링할 수 있고, Project 대시보드 Overview 섹션에서 언제든 태그를 추가/수정할 수 있습니다.
6. **여러 Run 세트로 experiment 비교**: 여러 Run 세트를 만들면 다양한 메트릭을 한 번에 비교할 수 있습니다. 차트별로 Run 세트의 표시/숨김도 자연스럽게 지원됩니다.

아래 코드 예제는 위에서 소개한 모범 사례를 모두 반영해 W&B Experiment 를 설정하는 방법을 보여줍니다:

```python
import wandb

config = {
    "learning_rate": 0.01,
    "momentum": 0.2,
    "architecture": "CNN",
    "dataset_id": "cats-0192",
}

with wandb.init(
    project="detect-cats",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
) as run:
    ...
```

W&B Experiment 정의 시 적용할 수 있는 파라미터에 관한 자세한 내용은 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}}) API 문서나 [API Reference Guide]({{< relref path="/ref/python/" lang="ko" >}})를 참고하세요.
