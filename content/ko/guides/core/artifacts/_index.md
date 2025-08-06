---
title: Artifacts
description: W&B Artifacts의 개요, 동작 방식, 그리고 시작 방법 안내.
cascade:
- url: guides/artifacts/:filename
menu:
  default:
    identifier: ko-guides-core-artifacts-_index
    parent: core
url: guides/artifacts
weight: 1
---

{{< cta-button productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifact_fundamentals.ipynb" >}}

W&B Artifacts를 사용하면 [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})의 입력 및 출력 데이터로써 데이터를 추적하고 버전 관리를 할 수 있습니다. 예를 들어, 모델 트레이닝 run은 데이터셋을 입력으로 받아 학습된 모델을 출력으로 생성할 수 있습니다. run에 하이퍼파라미터, 메타데이터, 메트릭을 기록할 수 있고, Artifacts를 활용해 모델 학습에 사용된 데이터셋(입력)과 결과로 생성된 모델 체크포인트(출력)를 각각 별도의 아티팩트로 기록, 추적, 버전 관리할 수 있습니다.

## 유스 케이스
Artifacts는 전체 ML 워크플로우의 입력과 출력으로 다양하게 활용할 수 있습니다. 데이터셋, 모델, 기타 아티팩트를 가공/처리의 입력값으로 사용할 수 있습니다.

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| 유스 케이스             | 입력                                  | 출력                                   |
|------------------------|---------------------------------------|----------------------------------------|
| Model Training         | Dataset (training and validation data) | Trained Model                          |
| Dataset Pre-Processing | Dataset (raw data)                     | Dataset (pre-processed data)           |
| Model Evaluation       | Model + Dataset (test data)            | [W&B Table]({{< relref path="/guides/models/tables/" lang="ko" >}})    |
| Model Optimization     | Model                                  | Optimized Model                        |


{{% alert %}}
아래의 코드 조각들은 순서대로 실행하는 것이 좋습니다.
{{% /alert %}}

## 아티팩트 생성하기

아티팩트는 단 4줄의 코드로 생성할 수 있습니다:
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 생성합니다.
2. [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}) API로 아티팩트 오브젝트를 만듭니다.
3. 모델 파일이나 데이터셋 등 하나 이상의 파일을 해당 아티팩트 오브젝트에 추가합니다.
4. 아티팩트를 W&B에 기록합니다.

예를 들어, 아래 코드 조각은 `example_artifact`라는 아티팩트에 `dataset.h5` 파일을 기록하는 예시입니다:

```python
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# "my_data"라는 아티팩트 버전을 dataset 타입으로 dataset.h5의 데이터와 함께 기록합니다.
```

- 아티팩트의 `type`은 W&B 플랫폼에서 어떻게 표시되는지에 영향을 미칩니다. `type`을 지정하지 않으면 기본값은 `unspecified`입니다.
- 드롭다운의 각 항목은 서로 다른 `type` 파라미터 값을 뜻합니다. 위 코드에서는 아티팩트의 `type`이 `dataset`입니다.

{{% alert %}}
Amazon S3 버킷 등 외부 오브젝트 스토리지에 저장된 파일 또는 디렉터리를 참조로 추가하는 방법은 [외부 파일 추적]({{< relref path="./track-external-files.md" lang="ko" >}}) 문서를 참고하세요. 
{{% /alert %}}

## 아티팩트 다운로드하기
[`use_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ko" >}}) 메서드를 사용하면 run의 입력으로 쓸 아티팩트를 지정할 수 있습니다.

위의 코드 예시에 이어, 아래 코드블록은 `training_dataset` 아티팩트를 사용하는 방법을 보여줍니다:

```python
artifact = run.use_artifact(
    "training_dataset:latest"
)  # "my_data" 아티팩트를 사용하는 run 오브젝트를 반환합니다.
```
이렇게 하면 아티팩트 오브젝트가 반환됩니다.

반환된 오브젝트를 이용해 아티팩트의 모든 데이터를 다운로드할 수 있습니다:

```python
datadir = (
    artifact.download()
)  # 전체 `my_data` 아티팩트를 기본 디렉터리에 다운로드합니다.
```

{{% alert %}}
`root` [파라미터]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}})에 원하는 경로를 지정하면 특정 디렉터리로 아티팩트를 다운로드할 수 있습니다. 다양한 다운로드 방법과 추가 파라미터는 [아티팩트 다운로드 및 사용]({{< relref path="./download-and-use-an-artifact.md" lang="ko" >}}) 가이드를 참고하세요.
{{% /alert %}}


## 다음 단계
* 아티팩트의 [버전 관리]({{< relref path="./create-a-new-artifact-version.md" lang="ko" >}})와 [업데이트]({{< relref path="./update-an-artifact.md" lang="ko" >}}) 방법을 배워보세요.
* [automations]({{< relref path="/guides/core/automations/" lang="ko" >}})를 이용해 아티팩트 변경에 따라 다운스트림 워크플로우를 트리거하거나 Slack 채널로 알림을 보내는 방법을 알아보세요.
* 학습된 모델을 관리하는 공간인 [registry]({{< relref path="/guides/core/registry/" lang="ko" >}})에 대해 확인해 보세요.
* [Python SDK]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}})와 [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ko" >}}) 레퍼런스 가이드도 참고해 보세요.