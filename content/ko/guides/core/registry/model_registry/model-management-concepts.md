---
title: 모델 레지스트리 용어 및 개념
description: 모델 레지스트리 용어 및 개념
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-model-management-concepts
    parent: model-registry
weight: 2
---

다음 용어들은 W&B Model Registry의 핵심 구성 요소를 설명합니다: [*model version*]({{< relref path="#model-version" lang="ko" >}}), [*model artifact*]({{< relref path="#model-artifact" lang="ko" >}}), 그리고 [*registered model*]({{< relref path="#registered-model" lang="ko" >}}).

## Model version
model version은 단일 모델 체크포인트를 나타냅니다. model version은 실험 내에서 특정 시점의 모델 및 그 파일들의 스냅샷입니다.

model version은 훈련된 모델을 설명하는 불변의 데이터 및 메타데이터 디렉토리입니다. W&B는 모델 아키텍처나 학습된 파라미터를 나중에 저장(및 복원)할 수 있도록, model version에 필요한 파일들을 추가할 것을 권장합니다.

model version은 하나의 [model artifact]({{< relref path="#model-artifact" lang="ko" >}})에만 속합니다. model version은 0개 이상의 [registered model]({{< relref path="#registered-model" lang="ko" >}})에 속할 수 있습니다. model version은 model artifact에 기록된 순서대로 저장됩니다. 동일한 model artifact에 로그를 남긴 모델이 기존 model version과 내용이 다를 경우, W&B는 자동으로 새로운 model version을 생성합니다.

PyTorch(https://pytorch.org/tutorials/beginner/saving_loading_models.html) 및 Keras(https://www.tensorflow.org/guide/keras/save_and_serialize)와 같은 모델 라이브러리에서 제공하는 직렬화 프로세스를 통해 생성된 파일을 model version 내부에 저장하세요.

## Model alias

model alias는 모델 version을 의미 있는 식별자로 고유하게 식별하거나 참조할 수 있게 해주는 수정 가능한 문자열입니다. registered model의 하나의 version에만 alias를 할당할 수 있습니다. 이는 프로그래밍적으로 사용할 때 alias가 항상 하나의 version을 참조해야 하기 때문입니다. 또한 alias를 통해 champion, candidate, production 등 모델 상태를 지정할 수 있습니다.

실제 사용 예시로는 "best", "latest", "production", "staging" 등의 alias를 부여해 특별한 의미를 줄 수 있습니다.

예를 들어, 모델을 생성하고 `"best"`라는 alias를 할당했다고 가정해 봅시다. 아래 예시처럼 해당 model을 `run.use_model` 로 불러올 수 있습니다.

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model tags
model tag는 하나 이상의 registered model에 속하는 키워드 혹은 라벨입니다.

model tag를 사용하여 registered model을 카테고리별로 정리하거나, Model Registry의 검색창에서 해당 카테고리로 검색할 수 있습니다. model tag는 Registered Model Card의 상단에 표시됩니다. 예를 들어 머신러닝 태스크, 담당 팀, 우선순위 등으로 registered model을 그룹화할 때 사용할 수 있습니다. 동일한 model tag를 여러 registered model에 추가하여 그룹화할 수 있습니다.

{{% alert %}}
model tag는 registered model에 적용되어 모델을 그룹화하거나 쉽게 찾도록 돕는 라벨로, [model alias]({{< relref path="#model-alias" lang="ko" >}})와는 다릅니다. model alias는 특정한 모델 version을 프로그래밍적으로 불러오기 위한 고유 식별자 또는 별명입니다. Model Registry에서 태그로 태스크를 정리하는 방법은 [Organize models]({{< relref path="./organize-models.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}

## Model artifact
model artifact는 기록된 [model version]({{< relref path="#model-version" lang="ko" >}})들의 집합입니다. model version은 model artifact에 기록된 순서대로 저장됩니다.

model artifact는 하나 이상의 model version을 포함할 수 있습니다. model version이 기록되지 않았다면 비어 있을 수도 있습니다.

예를 들어 model artifact를 생성하고 model training 중에 주기적으로 체크포인트를 저장한다고 가정해 봅시다. 각각의 체크포인트는 고유한 [model version]({{< relref path="#model-version" lang="ko" >}})에 해당합니다. model training 및 체크포인트 저장 시 생성된 모든 model version은 트레이닝 스크립트 시작 시 만들었던 단일 model artifact에 저장됩니다.

아래 이미지는 세 개의 model version(v0, v1, v2)을 포함하는 model artifact의 예시입니다.

{{< img src="/images/models/mr1c.png" alt="Model registry concepts" >}}

[모델 artifact 예시를 여기서 확인하세요](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n).

## Registered model
registered model은 여러 model version의 포인터(링크)들의 모음입니다. registered model은 동일한 ML 태스크용 후보 모델들을 '북마크'처럼 모아놓은 폴더로 생각할 수 있습니다. 각각의 "북마크"는 [model artifact]({{< relref path="#model-artifact" lang="ko" >}})에 속한 [model version]({{< relref path="#model-version" lang="ko" >}})을 가리킵니다. [model tag]({{< relref path="#model-tags" lang="ko" >}})를 이용해 registered model을 그룹화할 수 있습니다.

registered model은 하나의 모델링 유스 케이스 또는 태스크에 대한 후보 모델들을 대표하는 경우가 많습니다. 예를 들어 사용하는 모델별로 이미지 분류 태스크에 대해 `ImageClassifier-ResNet50`, `ImageClassifier-VGG16`, `DogBreedClassifier-MobileNetV2`와 같이 각각의 registered model을 만들 수도 있습니다. model version은 registered model에 연결된 순서대로 버전 번호가 할당됩니다.

[Registered Model 예시를 여기서 확인하세요](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions).