---
description: Model Registry terms and concepts
displayed_sidebar: default
---

# 용어 및 개념

<head>
  <title>모델 레지스트리 용어 및 개념</title>
</head>

다음 용어들은 W&B 모델 레지스트리의 핵심 구성 요소를 설명합니다: [*모델 버전*](#model-version), [*모델 아티팩트*](#model-artifact), 그리고 [*등록된 모델*](#registered-model).

## 모델 버전
모델 버전은 단일 모델 체크포인트를 나타냅니다. 모델 버전은 실험 내의 모델과 파일이 특정 시점에 스냅샷인 것입니다.

모델 버전은 학습된 모델을 설명하는 데이터와 메타데이터의 변경 불가능한 디렉터리입니다. W&B는 나중에 모델 아키텍처와 학습된 파라미터를 저장(및 복원)할 수 있도록 모델 버전에 파일을 추가하도록 권장합니다.

모델 버전은 한 개, 그리고 오직 한 개의 [모델 아티팩트](#model-artifact)에 속합니다. 모델 버전은 [등록된 모델](#registered-model)에 속할 수도 있고 속하지 않을 수도 있습니다. 모델 버전은 로그된 순서대로 모델 아티팩트에 저장됩니다. W&B는 동일한 모델 아티팩트에 로그된 모델이 이전 모델 버전과 다른 내용을 갖는 것을 감지하면 새 모델 버전을 자동으로 생성합니다.

모델 버전 내에 파일을 저장하세요. 이 파일들은 모델링 라이브러리(예: [PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) 및 [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize))에서 제공하는 직렬화 프로세스로부터 생성됩니다.

## 모델 별칭

모델 별칭은 의미론적으로 관련된 식별자로 등록된 모델의 모델 버전을 고유하게 식별하거나 참조할 수 있게 해주는 변경 가능한 문자열입니다. 별칭은 프로그래밍적으로 사용될 때 고유한 버전을 참조해야 하므로 등록된 모델의 한 버전에만 별칭을 할당할 수 있습니다. 이는 또한 별칭을 사용하여 모델의 상태(챔피언, 후보, 프로덕션)를 캡처할 수 있게 합니다.

"best", "latest", "production", "staging"과 같은 별칭을 사용하여 모델 버전에 특별한 목적을 표시하는 것이 일반적입니다.

예를 들어, 모델을 생성하고 `"best"` 별칭을 할당한 경우, `run.use_model`을 사용하여 해당 특정 모델을 참조할 수 있습니다.

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## 모델 태그
모델 태그는 하나 이상의 등록된 모델에 속한 키워드 또는 레이블입니다.

모델 태그를 사용하여 등록된 모델을 카테고리로 구성하고 모델 레지스트리의 검색 표시줄에서 해당 카테고리를 검색하세요. 모델 태그는 등록된 모델 카드의 상단에 표시됩니다. ML 작업, 소유 팀 또는 우선 순위별로 등록된 모델을 그룹화하기 위해 사용할 수 있습니다. 동일한 모델 태그는 그룹화를 위해 여러 등록된 모델에 추가될 수 있습니다.

:::info
모델 태그는 등록된 모델에 그룹화 및 검색 가능성을 위해 적용되는 레이블로, [모델 별칭](#model-alias)과 다릅니다. 모델 별칭은 프로그래밍 방식으로 모델 버전을 가져오는 데 사용하는 고유 식별자 또는 별명입니다. 모델 레지스트리에서 작업을 구성하기 위해 태그를 사용하는 방법에 대해 자세히 알아보려면 [모델 구성](./organize-models.md)을 참조하세요.
:::

## 모델 아티팩트
모델 아티팩트는 로그된 [모델 버전](#model-version)의 모음입니다. 모델 버전은 로그된 순서대로 모델 아티팩트에 저장됩니다.

모델 아티팩트는 하나 이상의 모델 버전을 포함할 수 있습니다. 모델 버전이 로그되지 않으면 모델 아티팩트는 비어 있을 수 있습니다.

예를 들어, 모델 아티팩트를 생성한다고 가정해 봅시다. 모델 학습 중, 체크포인트 동안 모델을 주기적으로 저장합니다. 각 체크포인트는 자체 [모델 버전](#model-version)에 해당합니다. 모델 학습 및 체크포인트 저장 중에 생성된 모든 모델 버전은 학습 스크립트 시작 시 생성한 동일한 모델 아티팩트에 저장됩니다.

다음 이미지는 세 개의 모델 버전: v0, v1, v2를 포함하는 모델 아티팩트를 보여줍니다.

![](@site/static/images/models/mr1c.png)

[예시 모델 아티팩트 보기](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/mnist-zws7gt0n).

## 등록된 모델
등록된 모델은 모델 버전에 대한 포인터(링크)의 모음입니다. 등록된 모델을 동일한 ML 작업에 대한 후보 모델의 "북마크" 폴더로 생각할 수 있습니다. 등록된 모델의 각 "북마크"는 [모델 아티팩트](#model-artifact)에 속한 [모델 버전](#model-version)을 가리키는 포인터입니다. [모델 태그](#model-tags)를 사용하여 등록된 모델을 그룹화할 수 있습니다.

등록된 모델은 종종 단일 모델링 사용 사례 또는 작업을 위한 후보 모델을 나타냅니다. 예를 들어, 사용하는 모델에 기반한 다양한 이미지 분류 작업에 대해 등록된 모델을 생성할 수 있습니다: "ImageClassifier-ResNet50", "ImageClassifier-VGG16", "DogBreedClassifier-MobileNetV2" 등. 모델 버전은 등록된 모델에 연결된 순서대로 버전 번호가 할당됩니다.

[예시 등록된 모델 보기](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions).