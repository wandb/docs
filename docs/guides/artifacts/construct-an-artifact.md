---
title: Create an artifact
description: W&B 아티팩트를 생성하고 구성합니다. 아티팩트에 하나 이상의 파일이나 URI 참조를 추가하는 방법을 배우세요.
displayed_sidebar: default
---

W&B Python SDK를 사용하여 [W&B Runs](../../ref/python/run.md)에서 아티팩트를 구성하세요. [파일, 디렉토리, URI, 병렬 run에서 아티팩트를 추가](#add-files-to-an-artifact)할 수 있습니다. 파일을 아티팩트에 추가한 후, 아티팩트를 W&B 서버나 [자신의 개인 서버](../hosting/hosting-options/self-managed.md)에 저장하세요.

Amazon S3에 저장된 파일과 같은 외부 파일을 추적하는 방법에 대한 정보는 [외부 파일 추적](./track-external-files.md) 페이지를 참조하세요.

## 아티팩트 구성하는 방법

[W&B Artifact](../../ref/python/artifact.md)를 세 단계로 구성하세요:

### 1. `wandb.Artifact()`로 아티팩트 Python 객체 생성하기

아티팩트 객체를 생성하려면 [`wandb.Artifact()`](../../ref/python/artifact.md) 클래스를 초기화하세요. 다음 파라미터를 지정하세요:

* **이름**: 아티팩트의 이름을 지정하세요. 이름은 고유하고 설명적이며 기억하기 쉬워야 합니다. 아티팩트 이름은 W&B App UI에서 아티팩트를 식별할 때와 해당 아티팩트를 사용하고자 할 때 모두 사용됩니다.
* **유형**: 유형을 제공하세요. 유형은 간단하고 설명적이며 기계학습 파이프라인의 단일 단계에 해당해야 합니다. 일반적인 아티팩트 유형에는 `'dataset'` 또는 `'model'`이 포함됩니다.

:::tip
제공된 "이름"과 "유형"은 방향 그래프를 생성하는 데 사용됩니다. 이는 W&B App에서 아티팩트의 계보를 볼 수 있음을 의미합니다.

자세한 내용은 [아티팩트 그래프 탐색 및 트래버스](./explore-and-traverse-an-artifact-graph.md)를 참조하세요.
:::

:::caution
아티팩트는 이름이 같을 수 없습니다. 심지어 다른 유형 파라미터를 지정하더라도 마찬가지입니다. 즉, `'dataset'` 유형의 'cats'라는 이름의 아티팩트와 `'model'`이라는 동일한 이름의 다른 아티팩트를 생성할 수 없습니다.
:::

아티팩트 객체를 초기화할 때 설명과 메타데이터를 선택적으로 제공할 수 있습니다. 사용 가능한 속성과 파라미터에 대한 자세한 내용은 Python SDK Reference Guide의 [wandb.Artifact](../../ref/python/artifact.md) 클래스 정의를 참조하세요.

다음 예시는 데이터셋 아티팩트를 생성하는 방법을 보여줍니다:

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

위의 코드 조각의 문자열 인수를 자신만의 이름과 유형으로 대체하세요.

### 2. 아티팩트에 하나 이상의 파일 추가하기

아티팩트 메소드를 사용하여 파일, 디렉토리, 외부 URI 레퍼런스(Amazon S3 등)를 추가하세요. 예를 들어, 단일 텍스트 파일을 추가하려면 [`add_file`](../../ref/python/artifact.md#add_file) 메소드를 사용하세요:

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

여러 파일을 추가하려면 [`add_dir`](../../ref/python/artifact.md#add_dir) 메소드를 사용할 수도 있습니다. 파일을 추가하는 방법에 대한 자세한 내용은 [아티팩트 업데이트](./update-an-artifact.md)를 참조하세요.

### 3. 아티팩트를 W&B 서버에 저장하기

마지막으로, 아티팩트를 W&B 서버에 저장하세요. 아티팩트는 run에 연결됩니다. 그러므로 아티팩트를 저장하려면 run 객체의 [`log_artifact()`](../../ref/python/run#log_artifact) 메소드를 사용하세요.

```python
# W&B Run 생성. 'job-type'을 대체하세요.
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

W&B run 외부에서 아티팩트를 구성할 수도 있습니다. 자세한 내용은 [외부 파일 추적](./track-external-files)을 참조하세요.

:::caution
`log_artifact` 호출은 고성능 업로드를 위해 비동기적으로 수행됩니다. 이는 루프에서 아티팩트를 로그할 때 예상치 못한 동작을 초래할 수 있습니다. 예를 들어:

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... 아티팩트 a에 파일 추가 ...
    run.log_artifact(a)
```

아티팩트 버전 **v0**는 메타데이터에 0 인덱스가 있다고 보장되지 않습니다. 아티팩트는 임의의 순서로 로그될 수 있습니다.
:::

## 아티팩트에 파일 추가하기

다음 섹션에서는 다른 파일 유형 및 병렬 run에서 아티팩트를 구성하는 방법을 보여줍니다.

다음 예제에서는 여러 파일과 디렉토리 구조를 가진 프로젝트 디렉토리가 있다고 가정합니다:

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### 단일 파일 추가하기

다음 코드조각은 아티팩트에 단일, 로컬 파일을 추가하는 방법을 보여줍니다:

```python
# 단일 파일 추가
artifact.add_file(local_path="path/file.format")
```

예를 들어, 작업 로컬 디렉토리에 `'file.txt'`라는 파일이 있다고 가정합니다.

```python
artifact.add_file("path/file.txt")  # `file.txt' 로 추가됨
```

아티팩트는 다음의 내용을 가지게 됩니다:

```
file.txt
```

옵션으로, 아티팩트 내의 경로를 `name` 파라미터에 전달할 수도 있습니다.

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

아티팩트는 다음으로 저장됩니다:

```
new/path/file.txt
```

| API 호출                                                  | 결과 아티팩트 |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### 여러 파일 추가하기

다음 코드조각은 아티팩트에 전체 로컬 디렉토리를 추가하는 방법을 보여줍니다:

```python
# 디렉토리를 재귀적으로 추가
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

다음의 API 호출은 다음의 아티팩트 내용을 생성합니다:

| API 호출                                    | 결과 아티팩트                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### URI 레퍼런스 추가하기

URI에 W&B 라이브러리가 처리할 수 있는 스킴이 있는 경우, Artifacts는 재현성을 위해 체크섬 및 기타 정보를 추적합니다.

[`add_reference`](../../ref/python/artifact#add_reference) 메소드를 사용하여 아티팩트에 외부 URI 레퍼런스를 추가합니다. `uri` 문자열을 자신의 URI로 대체하세요. 옵션으로 아티팩트 내의 경로를 `name` 파라미터로 전달할 수 있습니다.

```python
# URI 참조 추가
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts 는 현재 다음과 같은 URI 스킴을 지원합니다:

* `http(s)://`: HTTP를 통해 엑세스 가능한 파일 경로. 서버가 `ETag` 및 `Content-Length` 응답 헤더를 지원하는 경우, Artifacts는 etag와 크기 메타데이터 형태로 체크섬을 추적합니다.
* `s3://`: S3의 오브젝트 또는 오브젝트 접두어 경로. 버킷이 오브젝트 버전 관리를 활성화한 경우, 아티팩트는 레퍼런스된 오브젝트에 대한 체크섬 및 버전 관리 정보를 추적합니다. 오브젝트 접두어는 접두어 아래의 오브젝트를 최대 10,000개까지 포함하도록 확장됩니다.
* `gs://`: GCS의 오브젝트 또는 오브젝트 접두어 경로. 버킷이 오브젝트 버전 관리를 활성화한 경우, 아티팩트는 레퍼런스된 오브젝트에 대한 체크섬 및 버전 관리 정보를 추적합니다. 오브젝트 접두어는 접두어 아래의 오브젝트를 최대 10,000개까지 포함하도록 확장됩니다.

다음의 API 호출은 다음과 같은 아티팩트를 생성합니다:

| API 호출                                                                      | 결과 아티팩트 내용                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 병렬 run에서 아티팩트에 파일 추가하기

대규모 데이터셋이나 분산 트레이닝의 경우, 여러 병렬 run이 단일 아티팩트에 기여해야 할 수 있습니다.

```python
import wandb
import time

# 데모를 위해 ray를 사용하여 run을 병렬로 시작합니다.
# 병렬 run을 원하는 방법으로 구성할 수 있습니다.
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 각 병렬 작성자 배치에는 고유한 그룹 이름이 있어야 합니다.
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    작성 작업입니다. 각 작성자는 아티팩트에 하나의 이미지를 추가합니다.
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # wandb 테이블에 데이터를 추가합니다. 이 경우 예제 데이터를 사용합니다.
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # 아티팩트의 폴더에 테이블을 추가합니다.
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # 아티팩트를 추가하거나 데이터를 추가하는 작업을 수행합니다.
        run.upsert_artifact(artifact)


# 병렬로 run을 시작합니다.
result_ids = [train.remote(i) for i in range(num_parallel)]

# 파일이 아티팩트에 추가되었는지를 확인하기 위해 모든 작성자를 조인합니다.
ray.get(result_ids)

# 모든 작성자가 완료되었으면, 아티팩트를 마무리하여 준비 상태로 표시합니다.
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # 테이블 폴더를 가리키는 "PartitionTable"을 생성하고 아티팩트에 추가합니다.
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # 아티팩트를 마무리하여, 이 버전에 대한 향후 "upserts"를 허용하지 않습니다.
    run.finish_artifact(artifact)
```