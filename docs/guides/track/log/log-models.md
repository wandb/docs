---
displayed_sidebar: default
---

# 모델 로깅

다음 가이드는 W&B 실행에 모델을 로깅하고 상호 작용하는 방법을 설명합니다.

:::tip
다음 API는 실험 트래킹 워크플로의 일부로 모델을 추적하는 데 유용합니다. 이 페이지에 나열된 API를 사용하여 실행에 빠르게 모델을 로깅할 수 있으며, 메트릭, 테이블, 미디어 및 기타 개체와 함께 사용하세요.

W&B는 다음과 같은 경우 [W&B 아티팩트](../../artifacts/intro.md) 사용을 권장합니다:
- 모델 외에도 데이터세트, 프롬프트 등과 같은 직렬화된 데이터의 다양한 버전을 생성하고 추적하고자 할 때.
- W&B에서 추적한 모델이나 기타 개체의 [계보 그래프](../../artifacts/explore-and-traverse-an-artifact-graph.md)를 탐색하고자 할 때.
- 이 메서드로 생성된 모델 아티팩트와 상호 작용하고자 할 때, 예를 들어 [속성 업데이트](../../artifacts/update-an-artifact.md) (메타데이터, 별칭, 설명)

W&B 아티팩트 및 고급 버전 관리 사용 사례에 대한 자세한 내용은 [아티팩트](../../artifacts/intro.md) 문서를 참조하세요.
:::

:::info
이 페이지에 설명된 API를 사용하는 엔드 투 엔드 예제를 보려면 이 [Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb)을 참조하세요.
:::

## W&B 실행에 모델 로깅
[`log_model`](../../../ref/python/run.md#log_model)를 사용하여 지정한 디렉터리 내의 콘텐츠가 포함된 모델 아티팩트를 로깅하세요. [`log_model`](../../../ref/python/run.md#log_model) 메서드는 또한 결과 모델 아티팩트를 W&B 실행의 출력으로 표시합니다.

모델을 W&B 실행의 입력이나 출력으로 표시하면 모델의 의존성과 모델의 연관성을 추적할 수 있습니다. W&B 앱 UI 내에서 모델의 계보를 볼 수 있습니다. 자세한 내용은 [아티팩트](../../artifacts/intro.md) 장 내의 [아티팩트 그래프 탐색 및 트래버스](../../artifacts/explore-and-traverse-an-artifact-graph.md) 페이지를 참조하세요.

모델 파일이 저장된 경로를 `path` 파라미터에 제공하세요. 경로는 로컬 파일, 디렉터리 또는 `s3://bucket/path`와 같은 외부 버킷에 대한 [참조 URI](../../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references)일 수 있습니다.

`<>`로 묶인 값을 꼭 자신의 것으로 바꾸세요.

```python
import wandb

# W&B 실행 초기화
run = wandb.init(project="<your-project>", entity="<your-entity>")

# 모델 로깅
run.log_model(path="<path-to-model>", name="<name>")
```

`name` 파라미터에 대해 모델 아티팩트의 이름을 선택적으로 제공할 수 있습니다. `name`이 지정되지 않은 경우, W&B는 입력 경로의 기본 이름 앞에 실행 ID를 붙인 이름을 사용합니다.

:::tip
모델의 `name`을 추적하세요. W&B가 할당하든, 당신이 할당하든, 모델의 이름을 알아야 [`use_model`](https://docs.wandb.ai/ref/python/run#use_model) 메서드로 모델 경로를 검색할 때 필요합니다.
:::

자세한 내용은 API 참조 가이드의 [`log_model`](../../../ref/python/run.md#log_model)을 참조하세요.

<details>

<summary>예시: 실행에 모델 로깅</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B 실행 초기화
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# 하이퍼파라미터
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

# 학습 알고리즘
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

# 모델 학습 설정
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# 모델 저장
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# W&B 실행에 모델 로깅
run.log_model(path=full_path, name="MNIST")
run.finish()
```

사용자가 `log_model`을 호출했을 때 `MNIST`라는 이름의 모델 아티팩트가 생성되었고 `model.h5` 파일이 모델 아티팩트에 추가되었습니다. 터미널이나 노트북은 모델이 로깅된 실행에 대한 정보를 출력합니다.

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>

## 로깅된 모델 다운로드 및 사용
[`use_model`](../../../ref/python/run.md#use_model) 함수를 사용하여 이전에 W&B 실행에 로깅된 모델 파일에 엑세스하고 다운로드하세요.

모델 파일을 검색하려는 모델 아티팩트의 이름을 제공합니다. 제공하는 이름은 기존에 로깅된 모델 아티팩트의 이름과 일치해야 합니다.

`log_model`로 파일을 처음 로깅할 때 `name`을 정의하지 않았다면, 기본 할당된 이름은 실행 ID로 시작하는 입력 경로의 기본 이름입니다.

`<>`로 묶인 다른 값을 꼭 자신의 것으로 바꾸세요:
 
```python
import wandb

# 실행 초기화
run = wandb.init(project="<your-project>", entity="<your-entity>")

# 모델 엑세스 및 다운로드. 다운로드된 아티팩트의 경로를 반환
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model](../../../ref/python/run.md#use_model) 함수는 다운로드된 모델 파일의 경로를 반환합니다. 이 모델을 나중에 연결하려면 이 경로를 추적하세요. 앞의 코드 조각에서 반환된 경로는 `downloaded_model_path`라는 변수에 저장됩니다.

<details>

<summary>예시: 로깅된 모델 다운로드 및 사용</summary>

예를 들어, 다음 코드 조각에서 사용자는 `use_model` API를 호출했습니다. 그들은 가져오려는 모델 아티팩트의 이름을 지정하고 버전/별칭도 제공했습니다. 그런 다음 API에서 반환된 경로를 `downloaded_model_path` 변수에 저장했습니다.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # 모델 버전에 대한 의미 있는 별칭 또는 식별자
model_artifact_name = "fine-tuned-model"

# 실행 초기화
run = wandb.init(project=project, entity=entity)
# 모델 엑세스 및 다운로드. 다운로드된 아티팩트의 경로를 반환
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

자세한 내용은 API 참조 가이드의 [`use_model`](../../../ref/python/run.md#use_model)을 참조하세요.

## 모델을 W&B 모델 레지스트리에 로깅하고 연결
[`link_model`](../../../ref/python/run.md#link_model) 메서드를 사용하여 W&B 실행에 모델 파일을 로깅하고 [W&B 모델 레지스트리](../../model_registry/intro.md)에 연결하세요. 등록된 모델이 없는 경우 `registered_model_name` 파라미터에 제공한 이름으로 새로운 모델을 생성합니다.

:::tip
모델을 연결하는 것은 팀의 중앙 집중식 모델 저장소에 모델을 '북마크'하거나 '게시'하는 것과 유사하다고 생각할 수 있습니다.

모델을 연결할 때, 해당 모델은 [모델 레지스트리](../../model_registry/intro.md)에 복제되지 않습니다. 해당 모델은 프로젝트에서 레지스트리로 이동되지도 않습니다. 연결된 모델은 프로젝트의 원본 모델을 가리키는 포인터입니다.

[모델 레지스트리](../../model_registry/intro.md)를 사용하여 태스크별로 최고의 모델을 정리하고, 모델 수명 주기를 관리하며, ML 수명 주기 전반에 걸쳐 쉽게 추적 및 감사를 가능하게 하고, 웹훅이나 작업을 사용하여 하류 작업을 [자동화](../../model_registry/automation.md)하세요.
:::

*등록된 모델*은 [모델 레지스트리](../../model_registry/intro.md) 내의 연결된 모델 버전 모음 또는 폴더입니다. 등록된 모델은 일반적으로 단일 모델링 사용 사례 또는 태스크에 대한 후보 모델을 나타냅니다.

다음 코드 조각은 [`link_model`](../../../ref/python/run.md#link_model) API를 사용하여 모델을 연결하는 방법을 보여줍니다. `<>`로 묶인 다른 값을 꼭 자신의 것으로 바꾸세요:

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

자세한 내용은 API 참조 가이드의 [`link_model`](../../../ref/python/run.md#link_model)을 참조하세요.

`registered-model-name`이 모델 레지스트리 내에 이미 존재하는 등록된 모델의 이름과 일치하면, 모델은 해당 등록된 모델에 연결됩니다. 그러한 등록된 모델이 존재하지 않으면 새로운 모델이 생성되고 모델이 첫 번째로 연결됩니다.

예를 들어, 모델 레지스트리에 "Fine-Tuned-Review-Autocompletion"이라는 이름의 기존 등록된 모델이 있다고 가정해 보세요(예시 [여기](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) 참조). 그리고 몇 가지 모델 버전이 이미 연결되어 있다고 가정합니다: v0, v1, v2. `link_model`을 `registered-model-name="Fine-Tuned-Review-Autocompletion"`으로 호출하면 새 모델은 이 기존 등록된 모델에 v3으로 연결됩니다. 이 이름을 가진 등록된 모델이 존재하지 않으면 새로운 모델이 생성되고 새 모델이 v0으로 연결됩니다.


<details>

<summary>예시: W&B 모델 레지스트리에 모델 로깅 및 연결</summary>

예를 들어, 다음 코드 조각은 모델 파일을 로깅하고 모델을 등록된 모델 이름 `"Fine-Tuned-Review-Autocompletion"`에 연결합니다.

이 작업을 수행하기 위해 사용자는 `link_model` API를 호출합니다. API를 호출할 때 모델의 콘텐츠를 가리키는 로컬 파일 경로(`path`)와 연결할 등록된 모델의 이름(`registered_model_name`)을 제공합니다.

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

:::info
알림: 등록된 모델은 북마크된 모델 버전 모음을 보관합니다.
:::

</details>