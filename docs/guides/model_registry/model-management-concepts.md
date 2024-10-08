---
title: Model Registry Terms and Concepts
description: 모델 레지스트리 용어 및 개념
displayed_sidebar: default
---

다음 용어들은 W&B 모델 레지스트리의 핵심 구성 요소를 설명합니다: [*model version*](#model-version), [*model artifact*](#model-artifact), 그리고 [*registered model*](#registered-model).

## Model version
모델 버전은 단일 모델 체크포인트를 나타냅니다. 모델 버전은 실험 내의 모델과 해당 파일들의 특정 시점에 대한 스냅샷입니다.  

모델 버전은 훈련된 모델을 설명하는 데이터와 메타데이터의 불변 디렉토리입니다. W&B는 나중에 모델 아키텍처와 학습된 파라미터를 저장(및 복원)할 수 있는 파일을 모델 버전에 추가하는 것을 제안합니다.  

모델 버전은 하나의 [model artifact](#model-artifact)에만 속합니다. 모델 버전은 0개 이상의 [registered models](#registered-model)에 속할 수 있습니다. 모델 버전은 모델 artifact에 로그되는 순서대로 모델 artifact에 저장됩니다. W&B는 로그한 모델이 이전 모델 버전과 다른 내용을 가진 경우 새로운 모델 버전을 자동으로 생성합니다.

모델 라이브러리에서 제공하는 직렬화 프로세스에 의해 생성된 파일을 모델 버전에 저장하세요 (예: [PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html)와 [Keras](https://www.tensorflow.org/guide/keras/save_and_serialize)).


## Model alias

모델 에일리어스는 모델 버전을 고유하게 식별하거나 참조할 수 있게 하는 가변적인 문자열입니다. 하나의 registered model에 하나의 버전만 에일리어스를 할당할 수 있습니다. 이는 에일리어스가 프로그래밍적으로 사용될 때 고유한 버전을 참조해야 하기 때문입니다. 또한, 모델의 상태(챔피언, 후보, 프로덕션)를 캡처하기 위해 에일리어스를 사용할 수 있게 합니다.

일반적으로 "best", "latest", "production" 또는 "staging"과 같은 에일리어스는 특별한 목적을 가진 모델 버전을 표시하는 데 사용됩니다.

예를 들어, 모델을 생성하고 `"best"` 에일리어스를 할당한다고 가정해보세요. 해당 모델을 특정하게 `run.use_model`를 통해 참조할 수 있습니다.

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model tags
모델 태그는 하나 이상의 registered models에 속하는 키워드 또는 레이블입니다.

모델 태그를 사용하여 등록된 모델을 카테고리로 조직하고 모델 레지스트리의 검색창에서 해당 카테고리를 검색할 수 있습니다. 모델 태그는 Registered Model Card의 상단에 나타납니다. ML 작업이나 소유 팀, 우선순위에 따라 등록된 모델을 그룹화하기 위해 사용할 수 있습니다. 동일한 모델 태그는 여러 등록된 모델에 추가되어 그룹화가 가능하게 합니다.

:::info
모델 태그는 등록된 모델에 그룹화 및 검색 가능성을 위해 적용되는 레이블로, [model aliases](#model-alias)와 다릅니다. 모델 에일리어스는 모델 버전을 프로그래밍적으로 가져오는 데 사용하는 고유 식별자 또는 별명입니다. 태그를 사용하여 모델 레지스트리의 작업을 구성하는 방법에 대해 자세히 알아보려면 [Organize models](./organize-models.md)를 참조하세요.
:::


## Model artifact
모델 아티팩트는 기록된 [model versions](#model-version)의 컬렉션입니다. 모델 버전들은 모델 artifact에 로그되는 순서대로 저장됩니다.

모델 아티팩트는 하나 이상의 모델 버전을 포함할 수 있습니다. 어떤 모델 버전도 로그되지 않은 경우 모델 아티팩트는 비어 있을 수 있습니다.

예를 들어, 모델 아티팩트를 생성했다고 가정해보겠습니다. 모델 트레이닝 중, 체크포인트마다 주기적으로 모델을 저장합니다. 각 체크포인트는 자체 [model version](#model-version)에 해당됩니다. 모델 트레이닝 및 체크포인트 저장 중에 생성된 모든 모델 버전은 트레이닝 스크립트의 시작에 생성한 동일한 모델 아티팩트에 저장됩니다.

아래 이미지에서는 세 개의 모델 버전(v0, v1, v2)을 포함하는 모델 아티팩트를 보여줍니다.

![](/images/models/mr1c.png)

[여기에서 모델 아티팩트 예시를 확인하세요](https://wandb.ai/timssweeney/model_management_docs_official_v0/artifacts/model/mnist-zws7gt0n).

## Registered model
Registered model은 모델 버전에 대한 포인터(링크) 모음입니다. Registered model을 동일한 ML 작업에 대한 후보 모델의 "북마크" 폴더로 생각할 수 있습니다. Registered model의 각 "북마크"는 [model artifact](#model-artifact)에 속하는 [model version](#model-version)에 대한 포인터입니다. 등록된 모델을 그룹화하려면 [model tags](#model-tags)를 사용할 수 있습니다.

Registered models는 종종 단일 모델링 유스 케이스 또는 작업에 대한 후보 모델을 나타냅니다. 예를 들어, 사용하는 모델에 따라 다른 이미지 분류 작업을 위한 registered models를 생성할 수 있습니다: "ImageClassifier-ResNet50", "ImageClassifier-VGG16", "DogBreedClassifier-MobileNetV2" 등. 모델 버전은 registered model에 연결된 순서대로 버전 번호가 할당됩니다.

[여기에서 Registered Model 예시를 확인하세요](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions).