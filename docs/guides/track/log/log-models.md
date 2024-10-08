---
title: Log models
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb'/>

# Log models

다음 가이드는 모델을 W&B run에 로그하고 상호작용하는 방법을 설명합니다.

:::tip
다음 API들은 실험 추적 워크플로우의 일환으로 모델을 추적할 때 유용합니다. 이 페이지에 나열된 API를 사용하여 run에 모델을 빠르게 기록할 수 있으며, 메트릭, 테이블, 미디어 및 기타 오브젝트와 함께 기록할 수 있습니다.

W&B에서는 다음과 같은 경우 [W&B Artifacts](../../artifacts/intro.md)를 사용할 것을 권장합니다:
- 데이터셋, 프롬프트 등등 모델 외에 직렬화된 데이터의 다른 버전을 생성하고 추적합니다.
- W&B에서 추적한 모델이나 다른 오브젝트의 [계보 그래프](../../artifacts/explore-and-traverse-an-artifact-graph.md)를 탐색합니다.
- 이 메소드들로 생성된 모델 아티팩트와 상호작용합니다. 예를 들어 [속성 업데이트](../../artifacts/update-an-artifact.md) (메타데이터, 에일리어스, 설명) 등을 수행합니다.

W&B Artifacts 및 고급 버전 관리 유스 케이스에 대한 더 많은 정보는 [Artifacts](../../artifacts/intro.md) 문서를 참조하세요.
:::

## Log a model to a run
[`log_model`](../../../ref/python/run.md#log_model)를 사용하여 특정 디렉토리에 있는 내용을 포함하는 모델 아티팩트를 로그합니다. [`log_model`](../../../ref/python/run.md#log_model) 메소드는 결과 모델 아티팩트를 W&B run의 출력으로 표시합니다.

모델을 W&B run의 입력 또는 출력으로 표시하면 모델의 종속성 및 연관성을 추적할 수 있습니다. W&B 앱 UI에서 모델의 계보를 볼 수 있습니다. [Artifacts](../../artifacts/intro.md) 챕터 내의 [아티팩트 그래프 탐색 및 트래버스](../../artifacts/explore-and-traverse-an-artifact-graph.md) 페이지를 참조하여 자세한 정보를 얻으세요.

모델 파일이 저장된 경로를 `path` 파라미터에 제공합니다. 경로는 로컬 파일, 디렉토리 또는 `s3://bucket/path`와 같은 외부 버킷의 [참조 URI](../../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references)일 수 있습니다.

`<>`로 감싸진 값을 본인의 값으로 교체하세요.

import wandb

# Initialize a W&B run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Log the model
run.log_model(path="<path-to-model>", name="<name>")

옵션으로 모델 아티팩트의 이름을 `name` 파라미터로 제공할 수 있습니다. `name`이 지정되지 않으면 W&B는 입력 경로의 기본 이름 앞에 run ID를 붙인 이름을 사용합니다.

:::tip
당신이나 W&B가 모델에 지정한 `name`을 기록하세요. 나중에 모델 경로를 [`use_model`](/ref/python/run#use_model) 메소드를 사용하여 검색해야 할 때 이 이름이 필요합니다.
:::

가능한 파라미터에 대한 더 많은 정보는 API 참조 가이드 내의 [`log_model`](../../../ref/python/run.md#log_model)을 참조하세요.

<details>

<summary>예제: Log a model to a run</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# Initialize a W&B run
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# Hyperparameters
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

# Training algorithm
model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Configure the model for training
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Save model
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# Log the model to the W&B run
run.log_model(path=full_path, name="MNIST")
run.finish()
```

사용자가 `log_model`을 호출하면 `MNIST`라는 이름의 모델 아티팩트가 생성되며, 파일 `model.h5`가 해당 모델 아티팩트에 추가됩니다. 터미널 또는 노트북은 모델이 로그된 run에 대한 정보를 출력할 것입니다.

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>

## Download and use a logged model
이전에 W&B run에 로그된 모델 파일에 엑세스하고 다운로드하려면 [`use_model`](../../../ref/python/run.md#use_model) 함수를 사용하세요.

모델 파일이 저장된 모델 아티팩트의 이름을 제공합니다. 제공한 이름은 기존 로그된 모델 아티팩트의 이름과 일치해야 합니다.

처음 로그를 남길 때 `name`을 정의하지 않았다면, 기본적으로 할당된 이름은 입력 경로의 기본 이름에 run ID가 붙은 것입니다.

여기에 `<` 및 `>`로 감싸진 다른 값을 본인의 값으로 교체하세요.

```python
import wandb

# Initialize a run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model](../../../ref/python/run.md#use_model) 함수는 다운로드한 모델 파일의 경로를 반환합니다. 나중에 이 모델과 링크하려면 이 경로를 기록해 두세요. 앞선 코드조각에서 반환된 경로는 `downloaded_model_path`라는 변수에 저장됩니다.

<details>

<summary>예제: Download and use a logged model</summary>

예를 들어, 진행 중인 코드조각에서 사용자는 `use_model` API를 호출했습니다. 해당 API를 호출할 때 가져오려는 모델 아티팩트의 이름을 지정하고 버전/에일리어스도 제공했습니다. 그런 다음 API로부터 반환된 경로를 `downloaded_model_path` 변수에 저장했습니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전에 대한 의미 있는 닉네임 또는 식별자
model_artifact_name = "fine-tuned-model"

# Initialize a run
run = wandb.init(project=project, entity=entity)
# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

가능한 파라미터와 반환 유형에 대한 더 많은 정보는 API 참조 가이드 내의 [`use_model`](../../../ref/python/run.md#use_model)을 참조하세요.

## Log and link a model to the W&B Model Registry

:::info
[`link_model`](../../../ref/python/run.md#link_model) 메소드는 현재까진 구식 W&B 모델 레지스트리와만 호환되며, 이는 곧 사용 중지될 예정입니다. 모델 아티팩트를 새로운 모델 레지스트리 에디션에 연결하는 방법은 레지스트리 [문서](../../registry/link_version.md)를 방문해 확인해보세요.
:::

[`link_model`](../../../ref/python/run.md#link_model) 메소드를 사용해 모델 파일을 W&B run에 로그하고 [W&B 모델 레지스트리](../../model_registry/intro.md)로 연결할 수 있습니다. 등록된 모델이 없을 경우, W&B는 당신이 제공하는 `registered_model_name` 파라미터에 지정된 이름으로 새 모델을 만들어줄 것입니다.

:::tip
모델을 연결하는 것을 동료들이 볼 수 있는 팀의 중앙화된 모델 저장소에 모델을 '북마크하거나' '게시하기'로 생각할 수 있습니다.

모델을 연결한다는 것은 그 모델이 [모델 레지스트리](../../model_registry/intro.md)에 복제되거나 프로젝트 밖으로 이동하는 것이 아닙니다. 연결된 모델은 프로젝트 내의 원래 모델을 가리키는 포인터입니다.

[모델 레지스트리](../../model_registry/intro.md)를 사용해 작업별로 최상의 모델을 구성하고, 모델 생애 주기를 관리하며, ML 생애 주기 전체에서 추적 및 감사를 용이하게 하고, 웹훅 또는 작업을 통해 [후속 작업을 자동화](../../model_registry/model-registry-automations.md)합니다.
:::

등록 모델은 [모델 레지스트리](../../model_registry/intro.md) 내에서 연결된 모델 버전의 컬렉션 또는 폴더입니다. 일반적으로 등록된 모델은 단일 모델링 유스 케이스나 작업의 후보 모델을 나타냅니다.

다음 코드조각은 [`link_model`](../../../ref/python/run.md#link_model) API로 모델을 연결하는 방법을 보여줍니다. `<` 및 `>`로 감싸진 다른 값들을 본인의 값으로 교체하세요.

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

선택적으로 제공할 수 있는 파라미터에 대한 더 많은 정보는 API 참조 가이드 내의 [`link_model`](../../../ref/python/run.md#link_model)을 참조하세요.

`registered-model-name`과 모델 레지스트리에 이미 존재하는 등록 모델의 이름이 일치하면, 모델은 해당 등록된 모델과 연결됩니다. 해당 이름의 등록 모델이 존재하지 않으면 새 모델이 만들어지고 첫 번째 모델로 연결될 것입니다.

예를 들어, 모델 레지스트리에 "Fine-Tuned-Review-Autocompletion"이라는 기존 등록 모델이 있다고 가정해봅시다(예제 [여기](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)에서 확인하세요). 그리고 몇 가지 모델 버전들이 이미 연결되어 있다고 가정하세요: v0, v1, v2. `link_model`을 `registered-model-name="Fine-Tuned-Review-Autocompletion"`과 함께 호출하면 새로운 모델은 이 기존 등록 모델에 v3로 연결될 것입니다. 이 이름을 가진 등록된 모델이 없으면 새 모델이 생성되고 새로운 모델은 v0으로 연결됩니다.

<details>

<summary>예제: Log and link a model to the W&B Model Registry</summary>

예를 들어, 다음 코드조각은 모델 파일을 로그하고 모델을 `"Fine-Tuned-Review-Autocompletion"`이라는 이름의 등록 모델로 링크합니다.

이를 위해 사용자는 `link_model` API를 호출합니다. API를 호출할 때, 모델의 내용을 가리키는 로컬 파일 경로(`path`)를 제공하고, 이를 연결할 등록 모델의 이름을 제공합니다(`registered_model_name`).

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

:::info
알림: 등록된 모델은 북마크된 모델 버전들의 컬렉션을 보관하고 있습니다.
:::

</details>