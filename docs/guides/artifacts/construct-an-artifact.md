---
description: Create, construct a W&B Artifact. Learn how to add one or more files
  or a URI reference to an Artifact.
displayed_sidebar: default
---

# 아티팩트 구성하기

<head>
  <title>아티팩트 구성하기</title>
</head>

[W&B 실행](../../ref/python/run.md)에서 아티팩트를 구성하기 위해 W&B Python SDK를 사용하세요. [파일, 디렉터리, URI 및 병렬 실행에서 파일을 아티팩트에 추가할 수 있습니다](#add-files-to-an-artifact). 파일을 아티팩트에 추가한 후, 아티팩트를 W&B 서버 또는 [자체 개인 서버](../hosting/hosting-options/self-managed.md)에 저장하세요.

Amazon S3에 저장된 파일과 같은 외부 파일을 추적하는 방법에 대한 정보는 [외부 파일 추적](./track-external-files.md) 페이지를 참조하세요.

## 아티팩트 구성 방법

[W&B 아티팩트](../../ref/python/artifact.md)를 세 단계로 구성하세요:

### 1. `wandb.Artifact()`로 아티팩트 Python 객체 생성하기

[`wandb.Artifact()`](../../ref/python/artifact.md) 클래스를 초기화하여 아티팩트 객체를 생성하세요. 다음 파라미터를 지정하세요:

* **이름**: 아티팩트에 대한 이름을 지정하세요. 이름은 고유하며, 기억하기 쉽고 설명적이어야 합니다. 아티팩트 이름을 사용하여 W&B 앱 UI에서 아티팩트를 식별하고 해당 아티팩트를 사용할 때 모두 사용됩니다.
* **유형**: 유형을 제공하세요. 유형은 단순하고 설명적이며 머신 러닝 파이프라인의 단일 단계에 해당해야 합니다. 일반적인 아티팩트 유형에는 `'dataset'` 또는 `'model'`이 포함됩니다.


:::tip
제공한 "이름"과 "유형"은 방향성 비순환 그래프를 생성하는 데 사용됩니다. 이는 W&B 앱에서 아티팩트의 계보를 볼 수 있음을 의미합니다.

자세한 내용은 [아티팩트 그래프 탐색 및 트래버스](./explore-and-traverse-an-artifact-graph.md)를 참조하세요.
:::


:::caution
아티팩트는 유형 파라미터에 대해 다른 유형을 지정하더라도 동일한 이름을 가질 수 없습니다. 즉, 'dataset' 유형의 'cats'라는 이름의 아티팩트와 'model' 유형의 동일한 이름의 다른 아티팩트를 생성할 수 없습니다.
:::

아티팩트 객체를 초기화할 때 선택적으로 설명과 메타데이터를 제공할 수 있습니다. 사용 가능한 속성 및 파라미터에 대한 자세한 내용은 Python SDK 참조 가이드의 [wandb.Artifact](../../ref/python/artifact.md) 클래스 정의를 참조하십시오.

다음 예시는 데이터세트 아티팩트를 생성하는 방법을 보여줍니다:

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

앞서의 코드 조각에서 문자열 인수를 자신의 이름과 유형으로 대체하세요.

### 2. 아티팩트에 하나 이상의 파일 추가하기

artifact 메서드를 사용하여 파일, 디렉터리, 외부 URI 참조(예: Amazon S3) 등을 추가하세요. 예를 들어, 단일 텍스트 파일을 추가하려면 [`add_file`](../../ref/python/artifact.md#add_file) 메서드를 사용하세요:

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

[`add_dir`](../../ref/python/artifact.md#add_dir) 메서드를 사용하여 여러 파일을 추가할 수도 있습니다. 파일을 추가하는 방법에 대한 자세한 내용은 [아티팩트 업데이트](./update-an-artifact.md)를 참조하세요.

### 3. 아티팩트를 W&B 서버에 저장하기

마지막으로, 아티팩트를 W&B 서버에 저장하세요. 아티팩트는 실행과 연관되어 있으므로, 아티팩트를 저장하기 위해 실행 객체의 [`log_artifact()`](../../ref/python/run#log\_artifact) 메서드를 사용하세요.

```python
# W&B 실행을 생성합니다. 'job-type'을 대체하세요.
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

W&B 실행 외부에서 아티팩트를 구성할 수도 있습니다. 자세한 내용은 [외부 파일 추적](./track-external-files)을 참조하세요.

:::caution
`log_artifact` 호출은 성능 향상을 위해 비동기적으로 수행됩니다. 이로 인해 루프에서 아티팩트를 로깅할 때 예상치 못한 동작이 발생할 수 있습니다. 예를 들어:

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

아티팩트 버전 **v0**이 메타데이터에 인덱스 0을 가지고 있을 것이라는 보장은 없습니다. 아티팩트가 임의의 순서로 로깅될 수 있기 때문입니다.
:::

## 아티팩트에 파일 추가하기

다음 섹션에서는 다양한 파일 유형 및 병렬 실행에서 아티팩트를 구성하는 방법을 보여줍니다.

다음 예제의 경우, 여러 파일과 디렉터리 구조가 있는 프로젝트 디렉터리를 가지고 있다고 가정합니다:

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

다음 코드 조각은 아티팩트에 단일 로컬 파일을 추가하는 방법을 보여줍니다:

```python
# 단일 파일 추가하기
artifact.add_file(local_path="path/file.format")
```

예를 들어, 작업 로컬 디렉터리에 'file.txt'라는 파일이 있다고 가정해 보겠습니다.

```python
artifact.add_file("path/file.txt")  # 'file.txt'로 추가됨
```

아티팩트에는 이제 다음 내용이 포함되어 있습니다:

```
file.txt
```

`name` 파라미터에 아티팩트 내에서 원하는 경로를 선택적으로 전달할 수 있습니다.

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

아티팩트는 다음과 같이 저장됩니다:

```
new/path/file.txt
```

| API 호출                                                  | 결과 아티팩트 |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### 여러 파일 추가하기

다음 코드 조각은 아티팩트에 전체 로컬 디렉터리를 추가하는 방법을 보여줍니다:

```python
# 디렉터리 재귀적으로 추가하기
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

다음 API 호출은 다음과 같은 아티팩트 내용을 생성합니다:

| API 호출                                    | 결과 아티팩트                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### URI 참조 추가하기

W&B 라이브러리가 처리할 수 있는 스키마를 가진 URI의 경우, 아티팩트가 체크섬 및 기타 정보를 추적하여 재현성을 보장합니다.

아티팩트에 외부 URI 참조를 추가하려면 [`add_reference`](../../ref/python/artifact#add\_reference) 메서드를 사용하세요. `'uri'` 문자열을 자신의 URI로 대체하세요. name 파라미터에 아티팩트 내에서 원하는 경로를 선택적으로 전달할 수 있습니다.

```python
# URI 참조 추가하기
artifact.add_reference(uri="uri", name="optional-name")
```

아티팩트는 현재 다음 URI 스키마를 지원합니다:

* `http(s)://`: HTTP를 통해 접근할 수 있는 파일 경로. 아티팩트는 HTTP 서버가 `ETag` 및 `Content-Length` 응답 헤더를 지원하는 경우 etags 및 크기 메타데이터의 형태로 체크섬을 추적합니다.
* `s3://`: S3의 객체 또는 객체 접두사 경로. 아티팩트는 참조된 객체에 대한 체크섬 및 버전 관리 정보(버킷에 객체 버전 관리가 활성화된 경우)를 추적합니다. 객체 접두사는 최대 10,000개의 객체까지 접두사 아래에 있는 객체를 포함하도록 확장됩니다.
* `gs://`: GCS의 객체 또는 객체 접두사 경로. 아티팩트는 참조된 객체에 대한 체크섬 및 버전 관리 정보(버킷에 객체 버전 관리가 활성화된 경우)를 추적합니다. 객체 접두사는 최대 10,000개의 객체까지 접두사 아래에 있는 객체를 포함하도록 확장됩니다.

다음 API 호출은 다음과 같은 아티팩트를 생성합니다:

| API 호출                                                                      | 결과 아티팩트 내용                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 병렬 실행에서 아티팩트에 파일 추가하기

큰 데이터세트 또는 분산 학습의 경우, 여러 병렬 실행이 단일 아티팩트에 기여해야 할 수 있습니다.

```python
import wandb
import time

# 병렬 실행을 위해 ray를 사용합니다
# 시연 목적으로만 사용합니다. 병렬 실행을 원하는 방식으로 조정할 수 있습니다.
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 각 병렬 작성자 배치는 고유한 그룹 이름을 가져야 합니다.
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    작성자 작업. 각 작성자는 아티팩트에 하나의 이미지를 추가합니다.
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # 데이터를 wandb 테이블에 추가합니다. 이 경우 예제 데이터를 사용합니다
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # 테이블을 아티팩트의 폴더에 추가합니다
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # 아티팩트를 업서트하면 데이터를 아티팩트에 추가하거나 생성합니다
        run.upsert_artifact(artifact)


# 병렬 실행을 시작합니다
result_ids = [train.remote(i) for i in range(num_parallel)]

# 모든 작성자의 파일이 추가되었는지 확인하기 위해 모든 작성자에 조인합니다.
ray.get(result_ids)

# 모든 작성자가 완료되면, 아티팩트를 완성하여 준비 상태로 표시합니다.
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # 테이블 폴더를 가리키는 "PartitionTable"을 생성하고 아티팩트에 추가합니다.
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # 아티팩트를 완성하면 이 버전에 대한 추가 "업서트"가 허용되지 않습니다.
    run.finish_artifact(artifact)
```