---
title: Log models
menu:
  default:
    identifier: ko-guides-models-track-log-log-models
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# 모델 로깅

다음 가이드는 W&B run에 모델을 로깅하고 상호 작용하는 방법을 설명합니다.

{{% alert %}}
다음 API는 실험 트래킹 워크플로우의 일부로 모델을 추적하는 데 유용합니다. 이 페이지에 나열된 API를 사용하여 모델을 run에 기록하고 메트릭, 테이블, 미디어 및 기타 오브젝트에 엑세스합니다.

다음과 같은 경우 [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})를 사용하는 것이 좋습니다.
- 모델 외에 데이터셋, 프롬프트 등과 같은 직렬화된 데이터의 다양한 버전을 생성하고 추적합니다.
- W&B에서 추적된 모델 또는 다른 오브젝트의 [계보 그래프]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ko" >}})를 탐색합니다.
- [속성 업데이트]({{< relref path="/guides/core/artifacts/update-an-artifact.md" lang="ko" >}}) (메타데이터, 에일리어스 및 설명)와 같이 이러한 메서드가 생성한 모델 아티팩트와 상호 작용합니다.

W&B Artifacts 및 고급 버전 관리 유스 케이스에 대한 자세한 내용은 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 문서를 참조하십시오.
{{% /alert %}}

## 모델을 run에 로깅하기
지정한 디렉토리 내에 콘텐츠가 포함된 모델 아티팩트를 로깅하려면 [`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ko" >}})을 사용하십시오. [`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ko" >}}) 메서드는 결과 모델 아티팩트를 W&B run의 출력으로 표시합니다.

모델을 W&B run의 입력 또는 출력으로 표시하면 모델의 종속성 및 모델의 연결을 추적할 수 있습니다. W&B App UI 내에서 모델의 계보를 봅니다. 자세한 내용은 [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 챕터 내의 [아티팩트 그래프 탐색 및 트래버스]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ko" >}}) 페이지를 참조하십시오.

모델 파일이 저장된 경로를 `path` 파라미터에 제공합니다. 경로는 로컬 파일, 디렉토리 또는 `s3://bucket/path`와 같은 외부 버킷에 대한 [참조 URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ko" >}})일 수 있습니다.

`<>`로 묶인 값을 자신의 값으로 바꾸십시오.

```python
import wandb

# W&B run 초기화
run = wandb.init(project="<your-project>", entity="<your-entity>")

# 모델 로깅
run.log_model(path="<path-to-model>", name="<name>")
```

선택적으로 `name` 파라미터에 대한 모델 아티팩트 이름을 제공합니다. `name`이 지정되지 않은 경우 W&B는 run ID가 앞에 붙은 입력 경로의 기본 이름을 이름으로 사용합니다.

{{% alert %}}
사용자 또는 W&B가 모델에 할당한 `name`을 추적하십시오. [`use_model`]({{< relref path="/ref/python/run#use_model" lang="ko" >}}) 메서드로 모델 경로를 검색하려면 모델 이름이 필요합니다.
{{% /alert %}}

가능한 파라미터에 대한 자세한 내용은 API 참조 가이드의 [`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ko" >}})을 참조하십시오.

<details>

<summary>예제: 모델을 run에 로깅</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"옵티마이저": "adam", "loss": "categorical_crossentropy"}

# W&B run 초기화
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# 하이퍼파라미터
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

# 트레이닝 알고리즘
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

# 트레이닝을 위해 모델 구성
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# 모델 저장
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# 모델을 W&B run에 로깅
run.log_model(path=full_path, name="MNIST")
run.finish()
```

사용자가 `log_model`을 호출하면 `MNIST`라는 모델 아티팩트가 생성되고 파일 `model.h5`가 모델 아티팩트에 추가되었습니다. 터미널 또는 노트북에 모델이 로깅된 run에 대한 정보를 찾을 수 있는 위치에 대한 정보가 출력됩니다.

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>


## 로깅된 모델 다운로드 및 사용
[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ko" >}}) 함수를 사용하여 이전에 W&B run에 로깅된 모델 파일에 엑세스하고 다운로드합니다.

검색하려는 모델 파일이 저장된 모델 아티팩트의 이름을 제공합니다. 제공하는 이름은 기존에 로깅된 모델 아티팩트의 이름과 일치해야 합니다.

`log_model`로 파일을 원래 로깅할 때 `name`을 정의하지 않은 경우 할당된 기본 이름은 run ID가 앞에 붙은 입력 경로의 기본 이름입니다.

`<>`로 묶인 다른 값을 자신의 값으로 바꾸십시오.
 
```python
import wandb

# run 초기화
run = wandb.init(project="<your-project>", entity="<your-entity>")

# 모델에 엑세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다.
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model]({{< relref path="/ref/python/run.md#use_model" lang="ko" >}}) 함수는 다운로드된 모델 파일의 경로를 반환합니다. 나중에 이 모델을 연결하려는 경우 이 경로를 추적하십시오. 앞의 코드 조각에서 반환된 경로는 `downloaded_model_path`라는 변수에 저장됩니다.

<details>

<summary>예제: 로깅된 모델 다운로드 및 사용</summary>

예를 들어, 진행 중인 코드 조각에서 사용자는 `use_model` API를 호출했습니다. 그들은 가져오려는 모델 아티팩트의 이름을 지정하고 버전/에일리어스도 제공했습니다. 그런 다음 API에서 반환된 경로를 `downloaded_model_path` 변수에 저장했습니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전에 대한 시맨틱 닉네임 또는 식별자
model_artifact_name = "fine-tuned-model"

# run 초기화
run = wandb.init(project=project, entity=entity)
# 모델에 엑세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다.
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

가능한 파라미터 및 반환 유형에 대한 자세한 내용은 API 참조 가이드의 [`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ko" >}})을 참조하십시오.

## 모델을 로깅하고 W&B Model Registry에 연결

{{% alert %}}
[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ko" >}}) 메서드는 현재 곧 지원이 중단될 레거시 W&B Model Registry와만 호환됩니다. 모델 아티팩트를 새 버전의 모델 레지스트리에 연결하는 방법을 배우려면 레지스트리 [문서]({{< relref path="../../registry/link_version.md" lang="ko" >}})를 방문하십시오.
{{% /alert %}}

[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ko" >}}) 메서드를 사용하여 모델 파일을 W&B run에 로깅하고 [W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})에 연결합니다. 등록된 모델이 없으면 W&B는 `registered_model_name` 파라미터에 제공하는 이름으로 새 모델을 만듭니다.

{{% alert %}}
모델 연결은 팀의 다른 구성원이 보고 사용할 수 있는 모델의 중앙 집중식 팀 리포지토리에 모델을 '북마크'하거나 '게시'하는 것과 유사하다고 생각할 수 있습니다.

모델을 연결하면 해당 모델이 [Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})에서 복제되지 않습니다. 또한 해당 모델은 프로젝트에서 레지스트리로 이동되지 않습니다. 연결된 모델은 프로젝트의 원래 모델에 대한 포인터입니다.

[Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})를 사용하여 작업별로 가장 적합한 모델을 구성하고, 모델 라이프사이클을 관리하고, ML 라이프사이클 전반에 걸쳐 쉬운 추적 및 감사를 용이하게 하고, 웹훅 또는 작업을 통해 다운스트림 작업을 [자동화]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ko" >}})합니다.
{{% /alert %}}

*등록된 모델*은 [Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ko" >}})에서 연결된 모델 버전의 컬렉션 또는 폴더입니다. 등록된 모델은 일반적으로 단일 모델링 유스 케이스 또는 작업에 대한 후보 모델을 나타냅니다.

진행 중인 코드 조각은 [`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ko" >}}) API로 모델을 연결하는 방법을 보여줍니다. `<>`로 묶인 다른 값을 자신의 값으로 바꾸십시오.

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

선택적 파라미터에 대한 자세한 내용은 API 참조 가이드의 [`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ko" >}})을 참조하십시오.

`registered-model-name`이 Model Registry 내에 이미 존재하는 등록된 모델의 이름과 일치하면 모델이 해당 등록된 모델에 연결됩니다. 그러한 등록된 모델이 없으면 새 모델이 생성되고 모델이 처음 연결됩니다.

예를 들어, Model Registry에 "Fine-Tuned-Review-Autocompletion"이라는 기존 등록된 모델이 있다고 가정합니다(예제는 [여기](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) 참조). 그리고 몇 가지 모델 버전이 이미 v0, v1, v2에 연결되어 있다고 가정합니다. `registered-model-name="Fine-Tuned-Review-Autocompletion"`으로 `link_model`을 호출하면 새 모델이 이 기존 등록된 모델에 v3으로 연결됩니다. 이 이름의 등록된 모델이 없으면 새 모델이 생성되고 새 모델이 v0으로 연결됩니다.

<details>

<summary>예제: 모델을 로깅하고 W&B Model Registry에 연결</summary>

예를 들어, 진행 중인 코드 조각은 모델 파일을 로깅하고 모델을 등록된 모델 이름 `"Fine-Tuned-Review-Autocompletion"`에 연결합니다.

이렇게 하려면 사용자가 `link_model` API를 호출합니다. API를 호출할 때 모델의 콘텐츠를 가리키는 로컬 파일 경로(`path`)와 연결할 등록된 모델의 이름을 제공합니다(`registered_model_name`).

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
알림: 등록된 모델은 북마크된 모델 버전의 컬렉션을 보관합니다.
{{% /alert %}}

</details>
