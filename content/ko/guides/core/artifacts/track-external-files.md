---
title: 외부 파일 추적
description: 외부 버킷, HTTP 파일 서버 또는 NFS 공유에 저장된 파일을 추적합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**Reference artifact**를 사용하면 W&B 서버 외부(예: CoreWeave AI Object Storage, Amazon S3 버킷, GCS 버킷, Azure Blob, HTTP 파일 서버, NFS 공유 등)에 저장된 파일을 추적하고 활용할 수 있습니다.

W&B는 오브젝트의 ETag, 크기 등 오브젝트에 대한 메타데이터를 기록합니다. 버킷에 오브젝트 버전 관리가 활성화되어 있다면, 버전 ID도 함께 기록됩니다.

{{% alert %}}
외부 파일을 추적하지 않는 artifact를 로깅하면, W&B는 해당 artifact의 파일을 W&B 서버에 저장합니다. 이는 W&B Python SDK로 artifact를 기록할 때의 기본 동작입니다.

파일이나 디렉토리를 W&B 서버에 저장하는 방법은 [Artifacts 퀵스타트]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ko" >}})를 참고하세요.
{{% /alert %}}

아래는 reference artifact를 생성하는 방법을 설명합니다.

## 외부 버킷에서 artifact 추적하기

W&B Python SDK를 사용해 W&B 외부에 저장된 파일에 대한 reference를 추적합니다.

1. `wandb.init()`으로 run을 초기화합니다.
2. `wandb.Artifact()`로 artifact 오브젝트를 생성합니다.
3. artifact 오브젝트의 `add_reference()` 메소드로 버킷 경로를 지정합니다.
4. `run.log_artifact()`로 artifact의 메타데이터를 기록합니다.

```python
import wandb

# W&B run 초기화
run = wandb.init()

# artifact 오브젝트 생성
artifact = wandb.Artifact(name="name", type="type")

# 버킷 경로 reference 추가
artifact.add_reference(uri = "uri/to/your/bucket/path")

# artifact의 메타데이터 기록
run.log_artifact(artifact)
run.finish()
```

예를 들어, 다음과 같이 버킷이 구성되어 있다고 가정합니다:

```text
s3://my-bucket

|datasets/
  |---- mnist/
|models/
  |---- cnn/
```

`datasets/mnist/` 디렉토리에는 이미지 데이터가 저장되어 있습니다. 이 디렉토리를 `wandb.Artifact.add_reference()`로 데이터셋으로 추적할 수 있습니다. 아래 코드는 artifact 오브젝트의 `add_reference()` 메소드를 통해 reference artifact `mnist:latest`를 생성하는 예시입니다.

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact(name="mnist", type="dataset")
artifact.add_reference(uri="s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
run.finish()
```

W&B App에서 reference artifact의 파일 브라우저를 통해 내용을 살펴보고, [전체 dependency graph를 탐색]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ko" >}})하거나 artifact의 버전 히스토리를 조회할 수 있습니다. 다만, artifact 내부에 데이터 자체가 포함되어 있지 않기 때문에 이미지, 오디오 등 풍부한 미디어는 App에서 미리보기로 렌더링되지 않습니다.

{{% alert %}}
W&B Artifacts는 MinIO를 포함한 모든 Amazon S3 호환 인터페이스를 지원합니다. 아래에 설명된 스크립트는 `AWS_S3_ENDPOINT_URL` 환경변수를 MinIO 서버로 설정하면 MinIO에서도 바로 동작합니다.
{{% /alert %}}

{{% alert color="secondary" %}}
기본적으로 W&B는 객체 prefix 추가 시 1만 개(object) 제한을 둡니다. 이 제한을 변경하려면 `add_reference()` 호출 시 `max_objects=` 를 지정하세요.
{{% /alert %}}

## 외부 버킷에서 artifact 다운로드

W&B는 reference artifact를 다운로드할 때, artifact가 기록된 시점의 메타데이터를 활용해 버킷에서 파일을 가져옵니다. 버킷에 오브젝트 버전 관리가 활성화된 경우, artifact가 기록된 당시의 파일 상태에 해당하는 오브젝트 버전을 다운로드합니다. 버킷 내용을 지속적으로 관리하더라도, artifact가 버킷의 특정 시점(snapshot)을 캡처하기 때문에 항상 지정한 모델의 학습에 사용된 정확한 데이터 버전을 가리킬 수 있습니다.

다음 코드는 reference artifact를 다운로드하는 예시이며, reference artifact와 일반 artifact 모두 동일한 API를 사용합니다:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

{{% alert %}}
워크플로우에서 파일을 덮어쓰는 경우, 스토리지 버킷에 'Object Versioning'을 활성화하는 것을 권장합니다. 버전 관리가 활성화되어 있다면, 덮어쓴 파일의 이전 버전이 보존되어 있기 때문에 해당 파일을 참조하는 artifact도 유지됩니다. 

각 클라우드 환경에서 버전 관리를 활성화하는 방법은 아래에서 확인하세요: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/azure/storage/blobs/versioning-enable).
{{% /alert %}}

### 외부 reference 예제: 추가 및 다운로드

다음 코드는 데이터셋을 Amazon S3 버킷에 업로드하고, reference artifact로 추적한 뒤, 다운로드하는 예시입니다:

```python
import boto3
import wandb

run = wandb.init()

# 여기서 트레이닝 등 작업을 수행...

s3_client = boto3.client("s3")
s3_client.upload_file(file_name="my_model.h5", bucket="my-bucket", object_name="models/cnn/my_model.h5")

# 모델 artifact 로깅
model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

나중에, 모델 artifact를 다운로드할 수도 있습니다. artifact 이름과 타입을 지정하세요:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact(artifact_or_name = "cnn", type="model")
datadir = artifact.download()
```

{{% alert %}}
GCP 또는 Azure 환경에서 reference로 artifact를 추적하는 엔드투엔드 예시는 아래 Reports를 참고하세요:

* [Guide to Tracking Artifacts by Reference with GCP](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Working with Reference Artifacts in Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

## 클라우드 스토리지 인증 정보

W&B는 사용하는 클라우드 제공업체의 기본 인증 검색 방식을 사용합니다. 사용 중인 클라우드의 인증 정보 관련 문서를 참고하세요:

| 클라우드 제공업체         | 인증 정보 문서 |
| -------------- | ------------------------- |
| CoreWeave AI Object Storage | [CoreWeave AI Object Storage documentation](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys/cloud-console-tokens) |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS의 경우, 버킷이 현재 구성된 계정의 기본 리전에 존재하지 않는다면 `AWS_REGION` 환경변수 값을 버킷 리전에 맞게 지정해야 합니다.

{{% alert color="secondary" %}}
이미지, 오디오, 비디오, 포인트 클라우드 등과 같은 리치 미디어는 버킷의 CORS 설정에 따라 App UI에서 렌더링이 제한될 수 있습니다. 버킷의 CORS 설정에 **app.wandb.ai**를 Allow List에 추가하면 App UI에서 이러한 리치 미디어를 제대로 렌더링할 수 있습니다.

만약 이미지, 오디오, 비디오, 포인트 클라우드 등이 App UI에서 보이지 않는다면 버킷의 CORS 정책에 반드시 `app.wandb.ai`가 허용 목록에 포함되어 있는지 확인하세요.
{{% /alert %}}

## 파일 시스템에서 artifact 추적하기

데이터셋에 빠르게 접근하는 또 하나의 일반적인 방식은 NFS 마운트 포인트를 통해 모든 트레이닝 노드에서 접근 가능한 원격 파일 시스템을 노출하는 것입니다. 이 방법은 클라우드 스토리지 버킷보다 더 간단할 수 있습니다. 트레이닝 스크립트 입장에서는 파일이 로컬 파일 시스템에 있는 것처럼 보이기 때문입니다. 다행히도, Artifacts를 이용해 마운트 여부와 상관없이 파일 시스템 참조도 손쉽게 추적할 수 있습니다.

예를 들어, `/mount`에 다음과 같이 파일 시스템이 마운트되어 있다고 가정합니다:

```bash
mount
|datasets/
		|-- mnist/
|models/
		|-- cnn/
```

`mnist/` 경로에는 데이터셋(이미지 모음)이 존재합니다. 아래처럼 artifact로 추적할 수 있습니다:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
기본적으로 W&B는 디렉토리 reference 추가 시 1만 개 파일 제한을 둡니다. 제한을 변경하려면 `add_reference()` 호출 시 `max_objects=`를 지정하세요.
{{% /alert %}}

URL의 슬래시 3개에 주의하세요. 첫 번째는 파일 시스템 reference를 나타내는 `file://` prefix입니다. 다음은 데이터셋의 경로인 `/mount/datasets/mnist/`입니다.

이렇게 생성된 artifact `mnist:latest`는 일반 artifact처럼 보이고 동작합니다. 유일한 차이점은, artifact에는 파일 크기와 MD5 체크섬 등 메타데이터만 저장된다는 점입니다. 실제 파일 자체는 시스템 밖으로 전송되지 않습니다.

보통 artifact와 마찬가지로, App UI에서 reference artifact의 파일 브라우저, dependency graph, 버전 히스토리 등을 모두 탐색할 수 있습니다. 다만 artifact 내부에 데이터 자체가 없으므로 이미지나 오디오 등 리치 미디어는 UI에서 렌더링되지 않습니다.

reference artifact 다운로드 예시:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

파일 시스템 reference의 경우, `download()`를 실행하면 참조된 경로에서 파일을 복사해 artifact 디렉토리를 구성합니다. 위 예에서는 `/mount/datasets/mnist`의 내용이 `artifacts/mnist:v0/`로 복사됩니다. 만약 artifact에 참조된 파일이 덮어써져 사라졌다면, `download()`에서 에러가 발생할 수 있습니다.

위 과정을 하나로 합치면, 트레이닝 작업에 파일 시스템을 연결해 데이터셋을 추적하는 코드는 다음과 같습니다:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# artifact를 추적하며, 이 run의 입력으로 바로 표시합니다.
# 디렉토리 아래 파일이 변경되었을 때만 새 artifact 버전이 기록됩니다.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# 여기서 트레이닝 수행...
```

모델을 추적하는 경우, 트레이닝 스크립트가 모델 파일을 mount point에 저장한 후 모델 artifact를 기록하면 됩니다:

```python
import wandb

run = wandb.init()

# 트레이닝 진행...

# 디스크에 모델 저장

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```