---
title: Model Registry Terms and Concepts
description: 모델 레지스트리 용어 및 개념
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-model-management-concepts
    parent: model-registry
weight: 2
---

다음 용어는 W&B Model Registry의 주요 구성 요소를 설명합니다. [*모델 버전*]({{< relref path="#model-version" lang="ko" >}}), [*모델 아티팩트*]({{< relref path="#model-artifact" lang="ko" >}}) 및 [*등록된 모델*]({{< relref path="#registered-model" lang="ko" >}}).

## Model version
모델 버전은 단일 모델 체크포인트를 나타냅니다. 모델 버전은 실험 내에서 특정 시점의 모델과 해당 파일의 스냅샷입니다.

모델 버전은 학습된 모델을 설명하는 데이터 및 메타데이터의 변경 불가능한 디렉토리입니다. W&B는 모델 아키텍처와 학습된 파라미터를 나중에 저장하고 복원할 수 있도록 모델 버전에 파일을 추가할 것을 제안합니다.

모델 버전은 하나의 [model artifact]({{< relref path="#model-artifact" lang="ko" >}})에만 속합니다. 모델 버전은 0개 이상의 [registered models]({{< relref path="#registered-model" lang="ko" >}})에 속할 수 있습니다. 모델 버전은 모델 아티팩트에 기록된 순서대로 모델 아티팩트에 저장됩니다. W&B는 (동일한 model artifact에) 기록하는 모델이 이전 모델 버전과 다른 콘텐츠를 가지고 있음을 감지하면 자동으로 새 모델 버전을 생성합니다.

모델링 라이브러리에서 제공하는 직렬화 프로세스에서 생성된 파일을 모델 버전 내에 저장합니다(예: [PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html) 및 [Keras](https://www.tensorflow.org/guide/keras/save_and_serialize)).

## Model alias

모델 에일리어스는 등록된 모델에서 모델 버전을 의미적으로 관련된 식별자로 고유하게 식별하거나 참조할 수 있도록 하는 변경 가능한 문자열입니다. 에일리어스는 등록된 모델의 한 버전에만 할당할 수 있습니다. 이는 에일리어스가 프로그래밍 방식으로 사용될 때 고유한 버전을 참조해야 하기 때문입니다. 또한 에일리어스를 사용하여 모델의 상태(챔피언, 후보, production)를 캡처할 수 있습니다.

`"best"`, `"latest"`, `"production"` 또는 `"staging"`과 같은 에일리어스를 사용하여 특수 목적을 가진 모델 버전을 표시하는 것이 일반적입니다.

예를 들어 모델을 만들고 `"best"` 에일리어스를 할당한다고 가정합니다. `run.use_model`로 특정 모델을 참조할 수 있습니다.

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model tags
모델 태그는 하나 이상의 registered models에 속하는 키워드 또는 레이블입니다.

모델 태그를 사용하여 registered models를 카테고리로 구성하고 Model Registry의 검색 창에서 해당 카테고리를 검색합니다. 모델 태그는 Registered Model Card 상단에 나타납니다. ML 작업, 소유 팀 또는 우선 순위별로 registered models를 그룹화하는 데 사용할 수 있습니다. 그룹화를 위해 동일한 모델 태그를 여러 registered models에 추가할 수 있습니다.

{{% alert %}}
그룹화 및 검색 가능성을 위해 registered models에 적용되는 레이블인 모델 태그는 [model aliases]({{< relref path="#model-alias" lang="ko" >}})와 다릅니다. 모델 에일리어스는 모델 버전을 프로그래밍 방식으로 가져오는 데 사용하는 고유 식별자 또는 별칭입니다. 태그를 사용하여 Model Registry에서 작업을 구성하는 방법에 대한 자세한 내용은 [모델 구성]({{< relref path="./organize-models.md" lang="ko" >}})을 참조하세요.
{{% /alert %}}

## Model artifact
Model artifact는 기록된 [model versions]({{< relref path="#model-version" lang="ko" >}})의 모음입니다. 모델 버전은 모델 아티팩트에 기록된 순서대로 모델 아티팩트에 저장됩니다.

Model artifact는 하나 이상의 모델 버전을 포함할 수 있습니다. 모델 버전을 기록하지 않으면 Model artifact는 비어 있을 수 있습니다.

예를 들어, Model artifact를 만든다고 가정합니다. 모델 트레이닝 중에 체크포인트 중에 모델을 주기적으로 저장합니다. 각 체크포인트는 자체 [model version]({{< relref path="#model-version" lang="ko" >}})에 해당합니다. 모델 트레이닝 및 체크포인트 저장 중에 생성된 모든 모델 버전은 트레이닝 스크립트 시작 시 생성한 동일한 Model artifact에 저장됩니다.

다음 이미지는 v0, v1 및 v2의 세 가지 모델 버전을 포함하는 Model artifact를 보여줍니다.

{{< img src="/images/models/mr1c.png" alt="" >}}

[예제 Model artifact here](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n)를 봅니다.

## Registered model
Registered model은 모델 버전에 대한 포인터(링크) 모음입니다. Registered model을 동일한 ML 작업에 대한 후보 모델의 "북마크" 폴더라고 생각할 수 있습니다. Registered model의 각 "북마크"는 [model artifact]({{< relref path="#model-artifact" lang="ko" >}})에 속한 [model version]({{< relref path="#model-version" lang="ko" >}})에 대한 포인터입니다. [Model tags]({{< relref path="#model-tags" lang="ko" >}})를 사용하여 Registered models를 그룹화할 수 있습니다.

Registered models는 종종 단일 모델링 유스 케이스 또는 작업에 대한 후보 모델을 나타냅니다. 예를 들어 사용하는 모델을 기반으로 다양한 이미지 분류 작업에 대해 Registered model을 만들 수 있습니다. `ImageClassifier-ResNet50`, `ImageClassifier-VGG16`, `DogBreedClassifier-MobileNetV2` 등. 모델 버전은 Registered model에 연결된 순서대로 버전 번호가 할당됩니다.

[예제 Registered Model here](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions)를 봅니다.
