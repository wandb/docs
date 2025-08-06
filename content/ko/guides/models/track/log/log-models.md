---
title: 모델 로깅
menu:
  default:
    identifier: ko-guides-models-track-log-log-models
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# 모델 로깅

이 가이드에서는 W&B run 에 모델을 로깅하고 상호작용하는 방법을 설명합니다.

{{% alert %}}
아래의 API들은 실험 추적 워크플로우에서 모델을 추적하는 데 유용합니다. 이 페이지에 나열된 API를 활용하여 모델을 run 에 로깅하고, 메트릭, 테이블, 미디어, 기타 오브젝트에 엑세스하세요.

W&B에서는 모델 외에도 데이터셋, 프롬프트 등 여러 직렬화된 데이터 버전을 생성하거나 관리해야 한다면 [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 사용을 권장합니다.
- 모델, 데이터셋, 기타 오브젝트의 [계보 그래프]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ko" >}})를 탐색할 수 있습니다.
- 이 방법으로 생성된 모델 아티팩트의 [속성 업데이트]({{< relref path="/guides/core/artifacts/update-an-artifact.md" lang="ko" >}})(메타데이터, 에일리어스, 설명 등)도 가능합니다.

W&B Artifacts와 고급 버전 관리 유스 케이스에 대한 자세한 내용은 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 문서를 참고하세요.
{{% /alert %}}

## 모델을 run 에 로깅하기
[`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ko" >}}) 메소드를 이용해, 지정한 디렉토리 내의 내용이 포함된 모델 아티팩트를 run 에 로깅하세요. 이 메소드는 결과 모델 아티팩트를 해당 W&B run 의 output 으로도 마킹합니다.

모델을 run 의 input이나 output으로 마킹하면, 모델의 의존성과 연관관계를 추적할 수 있습니다. 모델의 계보(lineage)는 W&B App UI 에서 확인할 수 있으며, 자세한 내용은 [Artifacts 챕터의 계보 그래프 탐색]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ko" >}}) 문서를 참조하세요.

모델 파일이 저장된 경로를 `path` 파라미터에 전달하세요. 이 경로는 로컬 파일, 디렉토리, 또는 `s3://bucket/path` 같은 외부 버킷 URI([참조 URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ko" >}}))가 될 수 있습니다.

`<>`로 감싸진 값들은 본인의 정보로 반드시 교체하세요.

```python
import wandb

# W&B run 초기화
run = wandb.init(project="<your-project>", entity="<your-entity>")

# 모델 로깅
run.log_model(path="<path-to-model>", name="<name>")
```

옵션으로 `name` 파라미터에 모델 아티팩트의 이름을 지정할 수 있습니다. 만약 지정하지 않으면, W&B가 입력 경로의 베이스이름 앞에 run ID를 붙여서 기본 이름으로 사용합니다.

{{% alert %}}
직접 지정하거나 W&B가 자동 할당한 모델 `name` 값을 반드시 기록해 두세요. 이 이름은 나중에 [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ko" >}}) 메소드로 모델 경로를 불러올 때 필요합니다.
{{% /alert %}}

파라미터 상세는 API Reference의 [`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ko" >}}) 항목을 참고하세요.

<details>

<summary>예시: 모델을 run 에 로깅</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run 초기화
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# 하이퍼파라미터
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

# 트레이닝 알고리즘 구성
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

# 트레이닝용 모델 설정
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# 모델 저장
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# 모델을 W&B run 에 로깅
run.log_model(path=full_path, name="MNIST")
run.finish()
```

사용자가 `log_model`을 호출하면, `MNIST`라는 이름의 모델 아티팩트가 생성되고 `model.h5` 파일이 해당 아티팩트에 추가됩니다. 터미널 또는 노트북에는 모델이 로깅된 run 의 위치 정보가 출력됩니다.

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>


## 로깅된 모델 다운로드 및 활용
기존 W&B run 에 로깅된 모델 파일에 엑세스하고 다운로드하려면 [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ko" >}}) 함수를 사용하세요.

가져오려는 모델 파일이 저장된 모델 아티팩트의 이름을 꼭 전달해야 합니다. 입력한 이름과 동일한 이름으로 이미 로깅된 모델이 있어야만 맞게 동작합니다.

처음에 `log_model`로 파일을 저장할 때 `name`을 명시하지 않았다면, 경로의 베이스이름 앞에 run ID가 붙은 이름이 자동 지정됩니다.

기타 `<>`로 감싸진 값도 반드시 본인의 값으로 교체하세요.

```python
import wandb

# run 초기화
run = wandb.init(project="<your-project>", entity="<your-entity>")

# 모델 엑세스 및 다운로드. 다운로드 받은 아티팩트의 경로를 반환합니다.
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ko" >}}) 함수는 다운로드 받은 모델 파일 경로를 반환합니다. 나중에 해당 모델을 참조할 경우 이 경로를 잘 기록하세요. 위 코드조각에서는 반환된 경로가 `downloaded_model_path` 변수에 저장됩니다.

<details>

<summary>예시: 로깅된 모델 다운로드 및 활용</summary>

예를 들어, 다음 코드에서는 `use_model` API를 호출합니다. 가져오려는 모델 아티팩트의 이름과 버전/에일리어스를 지정하고, 반환된 경로를 `downloaded_model_path` 변수에 저장합니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전의 의미론적 별칭 또는 식별자
model_artifact_name = "fine-tuned-model"

# run 초기화
run = wandb.init(project=project, entity=entity)
# 모델 엑세스 및 다운로드. 다운로드 받은 아티팩트의 경로를 반환합니다.
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

파라미터 및 반환값 정보는 API Reference의 [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ko" >}}) 항목을 참조하세요.

## 모델 로깅 및 W&B Model Registry에 연결

{{% alert %}}
[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ko" >}}) 메소드는 현재 구 버전 W&B Model Registry에서만 사용할 수 있으며 곧 지원이 중단될 예정입니다. 새 Model Registry 에 모델 아티팩트를 연결하는 방법은 [Registry linking 가이드]({{< relref path="/guides/core/registry/link_version.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}

[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ko" >}}) 메소드를 사용하여 모델 파일을 W&B Run 에 로깅하고 [W&B Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ko" >}})에 연결할 수 있습니다. 등록 모델이 없을 경우, `registered_model_name` 파라미터로 입력한 이름으로 W&B가 새 모델을 생성합니다.

모델을 연결(linking)하는 것은 모델을 팀 중앙 저장소에 북마크하거나 공개하는 것과 비슷합니다. 팀의 다른 멤버가 이 모델을 쉽게 확인하고 사용할 수 있습니다.

모델을 연결할 때 모델이 [Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ko" >}})로 복제되거나 프로젝트에서 registry로 이동되는 것이 아닙니다. 연결된 모델은 프로젝트 내 원본 모델을 가리키는 포인터 역할입니다.

[Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})를 활용하면 과업별로 최고의 모델을 관리, 모델 라이프사이클 체계화, ML 전 과정에서의 추적·감사, 웹훅/잡을 통한 [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}}) 등이 쉬워집니다.

*Registered Model*은 [Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ko" >}}) 내에서 연결된 모델 버전들의 모음이나 폴더입니다. 일반적으로 단일 과업 또는 Use case의 후보 모델 그룹을 하나의 Registered Model로 관리합니다.

아래 코드 예시는 [`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ko" >}}) API로 모델을 연결하는 방법을 보여줍니다. `<>`로 된 부분은 반드시 본인 값으로 교체하세요.

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

선택 가능한 파라미터는 API Reference 가이드의 [`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ko" >}})를 참고하세요.

`registered-model-name`이 Model Registry 내 이미 존재하는 등록 모델 이름과 일치하면, 해당 등록 모델에 연결됩니다. 동일한 이름이 없으면 새 등록 모델이 생성되어 해당 모델이 첫 버전으로 연결됩니다.

예를 들어, Registry에 "Fine-Tuned-Review-Autocompletion"이라는 등록 모델([예시](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models))이 있고, v0, v1, v2 등이 이미 연결되어 있다고 가정해보겠습니다. 이때 `link_model`을 호출하면서 `registered-model-name="Fine-Tuned-Review-Autocompletion"`으로 지정하면, 새로운 모델은 v3으로 연결됩니다. 동일한 이름이 없다면, 새 등록 모델이 만들어지고 v0으로 연결됩니다.


<details>

<summary>예시: 모델 로깅 및 Model Registry에 연결</summary>

아래 예시에서는 모델 파일을 로깅하고 `"Fine-Tuned-Review-Autocompletion"`이라는 등록 모델 이름에 연결합니다.

사용자는 `link_model` API를 호출하면서, 모델이 들어있는 로컬 파일 경로 (`path`)와 연결할 등록 모델의 이름 (`registered_model_name`)을 제공합니다.

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
참고: 등록 모델(Registered Model)은 여러 모델 버전이 북마크된 컬렉션입니다.
{{% /alert %}}

</details>