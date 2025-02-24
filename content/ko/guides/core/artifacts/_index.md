---
title: Artifacts
description: W&B Artifacts 가 무엇인지, 어떻게 작동하는지, 그리고 W&B Artifacts 사용을 시작하는 방법에 대한 개요입니다.
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

W&B Artifacts 를 사용하여 [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 의 입력 및 출력으로 데이터를 추적하고 버전을 관리하세요. 예를 들어, 모델 트레이닝 run은 데이터셋을 입력으로 받아 트레이닝된 모델을 출력으로 생성할 수 있습니다. 하이퍼파라미터, 메타데이터 및 메트릭을 run에 기록하고 아티팩트를 사용하여 모델 트레이닝에 사용된 데이터셋을 입력으로, 결과 모델 체크포인트를 출력으로 기록, 추적 및 버전을 관리할 수 있습니다.

## 유스 케이스
[runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 의 입력 및 출력으로 ML 워크플로우 전체에서 아티팩트를 사용할 수 있습니다. 데이터셋, 모델 또는 다른 아티팩트를 처리하기 위한 입력으로 사용할 수 있습니다.

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| 유스 케이스               | 입력                       | 출력                       |
|------------------------|-----------------------------|------------------------------|
| Model Training         | Dataset (트레이닝 및 검증 데이터)     | Trained Model                |
| Dataset Pre-Processing | Dataset (raw data)          | Dataset (pre-processed data) |
| Model Evaluation       | Model + Dataset (테스트 데이터) | [W&B Table]({{< relref path="/guides/core/tables/" lang="ko" >}})                        |
| Model Optimization     | Model                       | Optimized Model              |


{{% alert %}}
다음 코드 조각은 순서대로 실행해야 합니다.
{{% /alert %}}

## 아티팩트 생성

다음 네 줄의 코드로 아티팩트를 생성합니다.
1. [W&B run]({{< relref path="/guides/models/track/runs/" lang="ko" >}}) 을 생성합니다.
2. [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ko" >}}) API를 사용하여 아티팩트 오브젝트를 생성합니다.
3. 모델 파일 또는 데이터셋과 같은 파일을 하나 이상 아티팩트 오브젝트에 추가합니다.
4. 아티팩트를 W&B에 기록합니다.

예를 들어, 다음 코드 조각은 `dataset.h5` 라는 파일을 `example_artifact` 라는 아티팩트에 기록하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(project = "artifacts-example", job_type = "add-dataset")
artifact = wandb.Artifact(name = "example_artifact", type = "dataset")
artifact.add_file(local_path = "./dataset.h5", name = "training_dataset")
artifact.save()

# "my_data" 아티팩트 버전을 dataset.h5 의 데이터가 있는 데이터셋으로 기록합니다.
```

{{% alert %}}
Amazon S3 버킷과 같은 외부 오브젝트 스토리지에 저장된 파일 또는 디렉토리에 대한 참조를 추가하는 방법에 대한 정보는 [track external files]({{< relref path="./track-external-files.md" lang="ko" >}}) 페이지를 참조하세요.
{{% /alert %}}

## 아티팩트 다운로드
[`use_artifact`]({{< relref path="/ref/python/run.md#use_artifact" lang="ko" >}}) 메소드를 사용하여 run에 대한 입력으로 표시할 아티팩트를 나타냅니다.

다음 코드 조각에 따라 다음 코드 블록은 `training_dataset` 아티팩트를 사용하는 방법을 보여줍니다.

```python
artifact = run.use_artifact("training_dataset:latest") # "my_data" 아티팩트를 사용하는 run 오브젝트를 반환합니다.
```
이렇게 하면 아티팩트 오브젝트가 반환됩니다.

다음으로 반환된 오브젝트를 사용하여 아티팩트의 모든 내용을 다운로드합니다.

```python
datadir = artifact.download() # 전체 "my_data" 아티팩트를 기본 디렉토리에 다운로드합니다.
```

{{% alert %}}
`root` [parameter]({{< relref path="/ref/python/artifact.md" lang="ko" >}}) 에 사용자 지정 경로를 전달하여 특정 디렉토리에 아티팩트를 다운로드할 수 있습니다. 아티팩트를 다운로드하는 다른 방법과 추가 파라미터를 보려면 [downloading and using artifacts]({{< relref path="./download-and-use-an-artifact.md" lang="ko" >}}) 가이드를 참조하세요.
{{% /alert %}}


## 다음 단계
* 아티팩트를 [version]({{< relref path="./create-a-new-artifact-version.md" lang="ko" >}}) 및 [update]({{< relref path="./update-an-artifact.md" lang="ko" >}}) 하는 방법을 알아봅니다.
* [artifact automation]({{< relref path="/guides/models/automations/project-scoped-automations/" lang="ko" >}}) 으로 아티팩트 변경에 대한 응답으로 다운스트림 워크플로우를 트리거하는 방법을 알아봅니다.
* 트레이닝된 모델을 보관하는 공간인 [registry]({{< relref path="/guides/models/registry/" lang="ko" >}}) 에 대해 알아봅니다.
* [Python SDK]({{< relref path="/ref/python/artifact.md" lang="ko" >}}) 및 [CLI]({{< relref path="/ref/cli/wandb-artifact/" lang="ko" >}}) 참조 가이드를 살펴봅니다.
