---
title: 아티팩트 생성
description: W&B 아티팩트를 생성하고 구성하는 방법을 알아보세요. 하나 이상의 파일이나 URI 참조를 아티팩트에 추가하는 방법을 배울
  수 있습니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK를 사용하여 [W&B Runs]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}})에서 Artifacts를 생성할 수 있습니다. [파일, 디렉토리, URI, 그리고 병렬 run에서 생성된 파일을 Artifacts에 추가]({{< relref path="#add-files-to-an-artifact" lang="ko" >}})할 수 있습니다. 파일을 Artifact에 추가한 후에는, Artifact를 W&B 서버 또는 [개인 서버]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}})에 저장하세요.

Amazon S3에 저장된 파일 등 외부 파일을 추적하는 방법은 [외부 파일 추적]({{< relref path="./track-external-files.md" lang="ko" >}}) 페이지를 참고하세요.

## Artifact 생성 방법

[W&B Artifact]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}})는 다음 세 단계로 생성합니다.

### 1. `wandb.Artifact()`로 Artifact Python 오브젝트 만들기

[`wandb.Artifact()`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}) 클래스를 초기화하여 Artifact 오브젝트를 생성합니다. 아래 파라미터들을 지정하세요.

* **Name**: Artifact의 이름을 지정하세요. 이 이름은 고유하고, 설명적이며 기억하기 쉬워야 합니다. Artifact의 이름은 W&B App UI에서 Artifact를 식별하거나, 해당 Artifact를 사용할 때 필요합니다.
* **Type**: 타입을 지정하세요. 이 타입은 간단하고, 설명적이며, 기계학습 파이프라인의 단계를 나타내야 합니다. 일반적으로 `'dataset'` 또는 `'model'`과 같은 타입이 사용됩니다.


{{% alert %}}
입력한 "name"과 "type"은 방향성 비순환 그래프(DAG)를 생성하는 데 사용됩니다. 즉, W&B App에서 Artifact의 계보를 시각화할 수 있습니다.

자세한 내용은 [Artifact 그래프 탐색 및 트래버스]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}


{{% alert color="secondary" %}}
같은 이름을 가진 Artifacts는 존재할 수 없습니다. type 파라미터에 서로 다른 값을 지정해도 마찬가지입니다. 즉, `cats`라는 이름의 타입이 `dataset`인 Artifact와, 동일한 이름의 타입이 `model`인 Artifact를 동시에 만들 수 없습니다.
{{% /alert %}}

Artifact 오브젝트 초기화 시 설명(description)과 메타데이터(metadata)도 옵션으로 제공할 수 있습니다. 사용 가능한 속성과 파라미터에 대한 자세한 내용은 Python SDK Reference의 [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}) 클래스를 참고하세요.

아래 예시에서는 데이터셋 Artifact를 만드는 방법을 보여줍니다.

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

위 코드조각의 문자열 인수들을 원하는 이름과 타입으로 바꿔 사용하세요.

### 2. 파일 하나 이상을 Artifact에 추가하기

Artifact 메소드를 활용하여 파일, 디렉토리, 외부 URI (예: Amazon S3) 등을 추가할 수 있습니다. 예를 들어, 텍스트 파일 하나를 추가하려면 [`add_file`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_file" lang="ko" >}}) 메소드를 사용합니다.

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

여러 파일을 추가하려면 [`add_dir`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_dir" lang="ko" >}}) 메소드를 사용할 수 있습니다. 파일 추가에 대한 자세한 내용은 [Artifact 업데이트]({{< relref path="./update-an-artifact.md" lang="ko" >}})를 참고하세요.

### 3. Artifact를 W&B 서버에 저장하기

마지막으로, Artifact를 W&B 서버에 저장하세요. Artifact는 Run과 연결되므로 Run 오브젝트의 [`log_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ko" >}}) 메소드를 사용하여 Artifact를 저장합니다.

```python
# W&B Run 생성 ('job-type'은 원하는 값으로 변경)
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

필요하다면, W&B Run 밖에서도 Artifact를 생성할 수 있습니다. 자세한 내용은 [외부 파일 추적]({{< relref path="./track-external-files.md" lang="ko" >}})을 참고하세요.

{{% alert color="secondary" %}}
`log_artifact` 호출은 성능 향상을 위해 비동기로 수행됩니다. 이로 인해, 반복문(loop)에서 Artifact를 로깅할 때 예상치 못한 동작이 발생할 수 있습니다. 예를 들어:

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... 여기서 파일을 artifact a에 추가 ...
    run.log_artifact(a)
```

Artifact 버전 **v0**의 메타데이터에서 index가 0이 된다는 보장이 없습니다. Artifact가 임의의 순서로 로깅될 수 있기 때문입니다.
{{% /alert %}}

## Artifact에 파일 추가하기

이 절에서는 파일 타입별, 병렬 Run에서 Artifacts를 생성하는 다양한 방법을 설명합니다.

다음 예제에서는 여러 파일 및 디렉토리가 위치한 프로젝트 디렉토리를 가정합니다.

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### 단일 파일 추가

아래 코드조각은 로컬 파일 하나를 Artifact에 추가하는 방법을 보여줍니다.

```python
# 파일 한 개 추가
artifact.add_file(local_path="path/file.format")
```

예를 들어, 작업 디렉토리에 `'file.txt'`라는 파일이 있다고 가정해 보겠습니다.

```python
artifact.add_file("path/file.txt")  # `file.txt`로 추가됨
```

Artifact는 이제 다음과 같이 구성됩니다.

```
file.txt
```

선택적으로, Artifact 내에서 원하는 경로를 `name` 파라미터에 지정할 수 있습니다.

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

이 경우, Artifact에는 다음과 같은 경로로 저장됩니다.

```
new/path/file.txt
```

| API 호출                                            | 결과 Artifact  |
| --------------------------------------------------- | -------------- |
| `artifact.add_file('model.h5')`                     | model.h5       |
| `artifact.add_file('checkpoints/model.h5')`         | model.h5       |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5 |

### 여러 파일 추가

아래 코드조각은 로컬 디렉토리 전체를 Artifact에 추가하는 방법을 보여줍니다.

```python
# 디렉토리를 재귀적으로 추가
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

아래의 API 호출들은 각각 다음과 같은 Artifact 콘텐츠를 생성합니다.

| API 호출                              | 결과 Artifact 내용                                     |
| ------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`          | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`      | `hello.txt`                                            |

### URI 참조 추가

W&B 라이브러리가 지원하는 스킴을 사용하는 URI의 경우, Artifacts는 재현성을 위해 체크섬 등 정보를 추적합니다.

Artifact에 외부 URI 참조를 추가하려면 [`add_reference`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_reference" lang="ko" >}}) 메소드를 사용하세요. `'uri'` 부분에는 본인의 URI를 입력하고, 필요하다면 Artifact 내부 경로를 name 파라미터로 전달하세요.

```python
# URI 참조 추가
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts가 현재 지원하는 URI 스킴은 다음과 같습니다.

* `http(s)://`: HTTP를 통해 접근 가능한 파일의 경로. HTTP 서버가 `ETag` 및 `Content-Length` 응답 헤더를 지원할 경우, Artifact에서 etag와 파일 크기 메타데이터로 체크섬을 추적합니다.
* `s3://`: S3 내 오브젝트 또는 오브젝트 프리픽스(object prefix)의 경로. 해당 오브젝트에 대해 체크섬과 버전 관리 정보(버킷이 오브젝트 버전 기능을 활성화해야 함)를 추적합니다. 오브젝트 프리픽스는 최대 10,000개 오브젝트까지 확장됩니다.
* `gs://`: GCS 내 오브젝트 또는 오브젝트 프리픽스의 경로. S3와 유사하게 체크섬과 버전 관리 정보를 추적하며, 오브젝트 프리픽스는 최대 10,000개까지 확장됩니다.

아래 API 호출들은 각각 다음과 같은 Artifacts를 생성합니다.

| API 호출                                                                  | 생성된 Artifact 내용                                      |
| ------------------------------------------------------------------------- | -------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                       | `model.h5`                                               |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`           | `model.h5`                                               |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                  |
| `artifact.add_reference('s3://my-bucket/images')`                         | <p><code>cat.png</code></p><p><code>dog.png</code></p>   |
| `artifact.add_reference('s3://my-bucket/images', name='images')`          | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 병렬 run에서 Artifact로 파일 추가하기

대규모 데이터셋이나 분산 트레이닝 시, 여러 병렬 run이 하나의 Artifact에 파일을 동시에 추가해야 할 수 있습니다.

```python
import wandb
import time

# 데모 목적을 위해 ray를 사용하여 run들을 병렬로 실행합니다.
# 병렬 run의 오케스트레이션 방법은 자유롭게 선택하면 됩니다.
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 병렬 writer 배치는 고유한 group 이름을 가져야 합니다.
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    writer 작업 예시. 각 writer는 Artifact에 이미지를 하나씩 추가합니다.
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # 예시 데이터로 wandb 테이블에 데이터 추가
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # 생성한 테이블을 Artifact의 폴더에 추가
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # upsert는 Artifact 생성 또는 데이터 추가를 의미
        run.upsert_artifact(artifact)


# 병렬로 run 실행
result_ids = [train.remote(i) for i in range(num_parallel)]

# 모든 writer의 파일이 Artifact에 추가될 때까지 대기
ray.get(result_ids)

# 모든 writer가 종료되면 Artifact를 finish 하여
# 사용 가능 상태로 만듭니다.
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # 폴더의 테이블들을 가리키는 "PartitionTable"을 생성하고
    # 이를 Artifact에 추가합니다.
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # finish_artifact는 해당 Artifact의 추가적인 "upsert"를
    # 이 버전에서는 막아 최종화합니다.
    run.finish_artifact(artifact)
```