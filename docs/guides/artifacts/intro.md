---
title: Artifacts
description: W&B Artifacts이 무엇인지, 어떻게 작동하는지, 그리고 W&B Artifacts를 사용하기 위한 시작 방법에 대한 개요.
slug: /guides/artifacts
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifact_fundamentals.ipynb"/>

W&B Artifacts를 사용하여 데이터를 [W&B Runs](../runs/intro.md)의 입력 및 출력으로서 추적하고 버전 관리할 수 있습니다. 예를 들어, 모델 트레이닝 run은 데이터셋을 입력으로 받아들이고, 트레이닝된 모델을 출력으로 생성할 수 있습니다. 하이퍼파라미터, 메타데이터, 메트릭을 run에 로그하는 것 외에도, 아티팩트를 사용하여 모델을 트레이닝하는 데 사용된 데이터셋을 입력으로, 그리고 결과 모델 체크포인트를 출력으로써 로그, 추적, 버전 관리할 수 있습니다.

## 유스 케이스
전체 ML 워크플로우에서 runs의 입력 및 출력으로 Artifacts를 사용할 수 있습니다. 데이터셋, 모델, 또는 심지어 다른 Artifacts를 처리의 입력으로 활용할 수 있습니다.

![](/images/artifacts/artifacts_landing_page2.png)

| 유스 케이스             | 입력                         | 출력                         |
|------------------------|-----------------------------|------------------------------|
| Model Training         | Dataset (training and validation data)     | Trained Model                |
| Dataset Pre-Processing | Dataset (raw data)          | Dataset (pre-processed data) |
| Model Evaluation       | Model + Dataset (test data) | [W&B Table](../tables/intro.md)                        |
| Model Optimization     | Model                       | Optimized Model              |


:::note
다음의 코드조각은 순서대로 실행해야 합니다.
:::

## 아티팩트 생성하기

네 줄의 코드를 통해 아티팩트를 생성하세요:
1. [W&B run](../runs/intro.md)을 생성합니다.
2. [`wandb.Artifact`](../../ref/python/artifact.md) API를 사용하여 아티팩트 오브젝트를 생성합니다.
3. 하나 이상의 파일, 예를 들어, 모델 파일이나 데이터셋을 아티팩트 오브젝트에 추가합니다.
4. 아티팩트를 W&B에 로그합니다.

예를 들어, 다음의 코드조각은 `example_artifact`라는 아티팩트에 `dataset.h5`라는 파일을 로그하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(project = "artifacts-example", job_type = "add-dataset")
artifact = wandb.Artifact(name = "example_artifact", type = "dataset")
artifact.add_file(local_path = "./dataset.h5", name = "training_dataset")
artifact.save()

# dataset.h5에서 데이터를 가져와 데이터셋으로서 아티팩트 버전 "my_data"를 로그합니다.
```

:::tip
Amazon S3 버킷처럼 외부 오브젝트 저장소에 저장된 파일이나 디렉토리에 대한 참조를 추가하는 방법은 [track external files](./track-external-files.md) 페이지를 참조하세요.
:::

## 아티팩트 다운로드
[`use_artifact`](../../ref/python/run.md#use_artifact) 메소드를 사용하여 run에 입력으로 표시할 아티팩트를 지정합니다.

이전 코드조각에 이어, 다음 코드는 `training_dataset` 아티팩트를 사용하는 방법을 보여줍니다: 

```python
artifact = run.use_artifact("training_dataset:latest") # "my_data" 아티팩트를 사용하는 run 오브젝트를 반환합니다.
```
이것은 아티팩트 오브젝트를 반환합니다.

다음으로, 반환된 오브젝트를 사용하여 아티팩트의 모든 내용을 다운로드합니다:

```python
datadir = artifact.download() # "my_data" 아티팩트를 기본 디렉토리에 전체 다운로드합니다.
```

:::tip
특정 디렉토리에 아티팩트를 다운로드하기 위해 `root` [파라미터](../../ref/python/artifact.md)에 사용자 정의 경로를 지정할 수 있습니다. 아티팩트를 다운로드하는 대체 방법과 추가 파라미터를 확인하려면 [downloading and using artifacts](./download-and-use-an-artifact.md) 가이드를 참조하세요.
:::

## 다음 단계
* 아티팩트를 [버전 관리](./create-a-new-artifact-version.md), [업데이트](./update-an-artifact.md), 또는 [삭제](./delete-artifacts.md)하는 방법을 배우세요.
* [artifact automation](./project-scoped-automations.md)을 통해 아티팩트 변경에 대응하여 다운스트림 워크플로우를 트리거하는 방법을 배우세요.
* 학습된 모델을 수용하는 공간인 [model registry](../model_registry/intro.md)에 대해 배워보세요.
* [Python SDK](../../ref/python/artifact.md)와 [CLI](../../ref/cli/wandb-artifact/README.md) 참조 가이드를 탐색하세요.