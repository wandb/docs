---
title: Create an artifact
description: W&B 아티팩트를 생성하고 구축합니다. 아티팩트에 하나 이상의 파일 또는 URI 참조를 추가하는 방법을 알아봅니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-construct-an-artifact
    parent: artifacts
weight: 2
---

W&B Python SDK를 사용하여 [W&B Runs]({{< relref path="/ref/python/run.md" lang="ko" >}})에서 Artifacts를 구성합니다. [파일, 디렉토리, URI 및 병렬 Runs의 파일을 Artifacts에 추가]({{< relref path="#add-files-to-an-artifact" lang="ko" >}})할 수 있습니다. Artifact에 파일을 추가한 후 Artifact를 W&B 서버 또는 [자체 개인 서버]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}})에 저장합니다.

Amazon S3에 저장된 파일과 같은 외부 파일을 추적하는 방법에 대한 자세한 내용은 [외부 파일 추적]({{< relref path="./track-external-files.md" lang="ko" >}}) 페이지를 참조하세요.

## Artifact를 구성하는 방법

다음 세 단계로 [W&B Artifact]({{< relref path="/ref/python/artifact.md" lang="ko" >}})를 구성합니다.

### 1. `wandb.Artifact()`로 아티팩트 Python 오브젝트를 생성합니다.

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ko" >}}) 클래스를 초기화하여 아티팩트 오브젝트를 생성합니다. 다음 파라미터를 지정합니다.

* **이름**: Artifact 이름을 지정합니다. 이름은 고유하고, 설명적이며, 기억하기 쉬워야 합니다. Artifact 이름을 사용하여 W&B App UI에서 Artifact를 식별하고, 해당 Artifact를 사용하려는 경우에도 Artifact 이름을 사용합니다.
* **유형**: 유형을 제공합니다. 유형은 간단하고 설명적이어야 하며, 기계 학습 파이프라인의 단일 단계를 나타내야 합니다. 일반적인 Artifact 유형에는 `'dataset'` 또는 `'model'`이 있습니다.

{{% alert %}}
제공하는 "이름"과 "유형"은 방향성 비순환 그래프를 만드는 데 사용됩니다. 즉, W&B App에서 Artifact의 계보를 볼 수 있습니다.

자세한 내용은 [Artifact 그래프 탐색 및 트래버스]({{< relref path="./explore-and-traverse-an-artifact-graph.md" lang="ko" >}})를 참조하세요.
{{% /alert %}}

{{% alert color="secondary" %}}
Artifacts는 types 파라미터에 대해 다른 유형을 지정하더라도 동일한 이름을 가질 수 없습니다. 즉, `dataset` 유형의 `cats`라는 Artifact와 `model` 유형의 동일한 이름을 가진 다른 Artifact를 만들 수 없습니다.
{{% /alert %}}

Artifact 오브젝트를 초기화할 때 선택적으로 설명과 메타데이터를 제공할 수 있습니다. 사용 가능한 속성 및 파라미터에 대한 자세한 내용은 Python SDK 참조 가이드에서 [`wandb.Artifact`]({{< relref path="/ref/python/artifact.md" lang="ko" >}}) 클래스 정의를 참조하세요.

다음 예제는 데이터셋 Artifact를 만드는 방법을 보여줍니다.

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

이전 코드 조각에서 문자열 인수를 자신의 이름과 유형으로 바꿉니다.

### 2. Artifact에 파일을 하나 이상 추가합니다.

Artifact 메소드를 사용하여 파일, 디렉토리, 외부 URI 참조 (예: Amazon S3) 등을 추가합니다. 예를 들어, 단일 텍스트 파일을 추가하려면 [`add_file`]({{< relref path="/ref/python/artifact.md#add_file" lang="ko" >}}) 메소드를 사용합니다.

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

[`add_dir`]({{< relref path="/ref/python/artifact.md#add_dir" lang="ko" >}}) 메소드를 사용하여 여러 파일을 추가할 수도 있습니다. 파일을 추가하는 방법에 대한 자세한 내용은 [Artifact 업데이트]({{< relref path="./update-an-artifact.md" lang="ko" >}})를 참조하세요.

### 3. Artifact를 W&B 서버에 저장합니다.

마지막으로, Artifact를 W&B 서버에 저장합니다. Artifacts는 Run과 연결되어 있습니다. 따라서 Run 오브젝트의 [`log_artifact()`]({{< relref path="/ref/python/run.md#log_artifact" lang="ko" >}}) 메소드를 사용하여 Artifact를 저장합니다.

```python
# W&B Run을 생성합니다. 'job-type'을 바꿉니다.
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

선택적으로 W&B Run 외부에서 Artifact를 구성할 수 있습니다. 자세한 내용은 [외부 파일 추적]({{< relref path="./track-external-files.md" lang="ko" >}})을 참조하세요.

{{% alert color="secondary" %}}
`log_artifact`에 대한 호출은 성능이 좋은 업로드를 위해 비동기적으로 수행됩니다. 이로 인해 루프에서 Artifacts를 로깅할 때 예기치 않은 동작이 발생할 수 있습니다. 예를 들어:

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... artifact a에 파일 추가 ...
    run.log_artifact(a)
```

Artifact 버전 **v0**은 Artifacts가 임의의 순서로 로깅될 수 있으므로 메타데이터에 인덱스 0이 있다고 보장되지 않습니다.
{{% /alert %}}

## Artifact에 파일 추가

다음 섹션에서는 다양한 파일 형식과 병렬 Runs에서 Artifacts를 구성하는 방법을 보여줍니다.

다음 예제에서는 여러 파일과 디렉토리 구조가 있는 프로젝트 디렉토리가 있다고 가정합니다.

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

다음 코드 조각은 단일 로컬 파일을 Artifact에 추가하는 방법을 보여줍니다.

```python
# 단일 파일 추가
artifact.add_file(local_path="path/file.format")
```

예를 들어, 작업 로컬 디렉토리에 `'file.txt'`라는 파일이 있다고 가정합니다.

```python
artifact.add_file("path/file.txt")  # `file.txt`로 추가됨
```

이제 Artifact에 다음 콘텐츠가 있습니다.

```
file.txt
```

선택적으로 `name` 파라미터에 대해 Artifact 내에서 원하는 경로를 전달합니다.

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

Artifact는 다음과 같이 저장됩니다.

```
new/path/file.txt
```

| API 호출                                                  | 결과 Artifact |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### 여러 파일 추가

다음 코드 조각은 전체 로컬 디렉토리를 Artifact에 추가하는 방법을 보여줍니다.

```python
# 디렉토리를 재귀적으로 추가합니다.
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

다음 API 호출은 다음 Artifact 콘텐츠를 생성합니다.

| API 호출                                    | 결과 Artifact                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### URI 참조 추가

Artifacts는 W&B 라이브러리가 처리 방법을 알고 있는 스키마가 URI에 있는 경우 재현성을 위해 체크섬 및 기타 정보를 추적합니다.

[`add_reference`]({{< relref path="/ref/python/artifact.md#add_reference" lang="ko" >}}) 메소드를 사용하여 외부 URI 참조를 Artifact에 추가합니다. `'uri'` 문자열을 자신의 URI로 바꿉니다. 선택적으로 name 파라미터에 대해 Artifact 내에서 원하는 경로를 전달합니다.

```python
# URI 참조 추가
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts는 현재 다음 URI 스키마를 지원합니다.

* `http(s)://`: HTTP를 통해 액세스할 수 있는 파일의 경로입니다. Artifact는 HTTP 서버가 `ETag` 및 `Content-Length` 응답 헤더를 지원하는 경우 etag 형식의 체크섬과 크기 메타데이터를 추적합니다.
* `s3://`: S3의 오브젝트 또는 오브젝트 접두사의 경로입니다. Artifact는 참조된 오브젝트에 대해 체크섬 및 버전 관리 정보(버킷에 오브젝트 버전 관리가 활성화된 경우)를 추적합니다. 오브젝트 접두사는 최대 10,000개의 오브젝트까지 접두사 아래의 오브젝트를 포함하도록 확장됩니다.
* `gs://`: GCS의 오브젝트 또는 오브젝트 접두사의 경로입니다. Artifact는 참조된 오브젝트에 대해 체크섬 및 버전 관리 정보(버킷에 오브젝트 버전 관리가 활성화된 경우)를 추적합니다. 오브젝트 접두사는 최대 10,000개의 오브젝트까지 접두사 아래의 오브젝트를 포함하도록 확장됩니다.

다음 API 호출은 다음 Artifacts를 생성합니다.

| API 호출                                                                      | 결과 Artifact 콘텐츠                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### 병렬 Runs에서 Artifacts에 파일 추가

대규모 데이터셋 또는 분산 트레이닝의 경우 여러 병렬 Runs가 단일 Artifact에 기여해야 할 수 있습니다.

```python
import wandb
import time

# 데모 목적으로 ray를 사용하여 Runs를 병렬로 시작합니다.
# 원하는 방식으로 병렬 Runs를 오케스트레이션할 수 있습니다.
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# 병렬 작성기의 각 배치는 자체 고유한 그룹 이름을 가져야 합니다.
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    작성기 작업입니다. 각 작성기는 Artifact에 이미지를 하나 추가합니다.
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # 데이터를 wandb 테이블에 추가합니다. 이 경우 예제 데이터를 사용합니다.
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # Artifact의 폴더에 테이블을 추가합니다.
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # Artifact를 업서트하면 Artifact에 데이터를 만들거나 추가합니다.
        run.upsert_artifact(artifact)


# 병렬로 Runs를 시작합니다.
result_ids = [train.remote(i) for i in range(num_parallel)]

# 모든 작성기가 완료되었는지 확인하기 위해 모든 작성기에 조인합니다.
# Artifact를 완료하기 전에 파일이 추가되었습니다.
ray.get(result_ids)

# 모든 작성기가 완료되면 Artifact를 완료합니다.
# 준비되었음을 표시합니다.
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # 테이블 폴더를 가리키는 "PartitionTable"을 만듭니다.
    # Artifact에 추가합니다.
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # Finish artifact는 Artifact를 완료하고 향후 "upserts"를 허용하지 않습니다.
    # 이 버전으로.
    run.finish_artifact(artifact)
```