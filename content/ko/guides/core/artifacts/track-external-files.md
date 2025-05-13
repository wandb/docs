---
title: Track external files
description: Amazon S3 버킷, GCS 버킷, HTTP 파일 서버 또는 NFS 공유와 같이 W&B 외부에 저장된 파일을 추적합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**reference artifacts** 를 사용하여 W&B 시스템 외부 (예: Amazon S3 버킷, GCS 버킷, Azure Blob, HTTP 파일 서버 또는 NFS 공유)에 저장된 파일을 추적합니다. W&B [CLI]({{< relref path="/ref/cli" lang="ko" >}})를 사용하여 [W&B Run]({{< relref path="/ref/python/run" lang="ko" >}}) 외부에서 아티팩트를 기록합니다.

### Run 외부에서 아티팩트 기록

W&B는 run 외부에서 아티팩트를 기록할 때 run을 생성합니다. 각 아티팩트는 run에 속하며, run은 프로젝트에 속합니다. 아티팩트 (버전)는 컬렉션에도 속하며, 유형이 있습니다.

[`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put" lang="ko" >}}) 코맨드를 사용하여 W&B run 외부의 W&B 서버에 아티팩트를 업로드합니다. 아티팩트가 속할 프로젝트 이름과 아티팩트 이름 (`project/artifact_name`)을 제공합니다. 선택적으로 유형 (`TYPE`)을 제공합니다. 아래 코드 조각에서 `PATH`를 업로드할 아티팩트의 파일 경로로 바꿉니다.

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

지정한 프로젝트가 존재하지 않으면 W&B가 새 프로젝트를 생성합니다. 아티팩트 다운로드 방법에 대한 자세한 내용은 [아티팩트 다운로드 및 사용]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ko" >}})을 참조하세요.

## W&B 외부에서 아티팩트 추적

데이터셋 버전 관리 및 모델 이력에 W&B Artifacts를 사용하고, **reference artifacts** 를 사용하여 W&B 서버 외부에서 저장된 파일을 추적합니다. 이 모드에서 아티팩트는 URL, 크기 및 체크섬과 같은 파일에 대한 메타데이터만 저장합니다. 기본 데이터는 시스템을 벗어나지 않습니다. 파일을 W&B 서버에 저장하는 방법에 대한 자세한 내용은 [빠른 시작]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ko" >}})을 참조하세요.

다음은 reference artifacts 를 구성하는 방법과 이를 워크플로우에 통합하는 가장 좋은 방법을 설명합니다.

### Amazon S3 / GCS / Azure Blob Storage 참조

클라우드 스토리지 버킷에서 참조를 추적하기 위해 데이터셋 및 모델 버전 관리에 W&B Artifacts를 사용합니다. 아티팩트 참조를 사용하면 기존 스토리지 레이아웃을 수정하지 않고도 버킷 위에 원활하게 추적 기능을 레이어링할 수 있습니다.

Artifacts는 기본 클라우드 스토리지 공급 업체 (예: AWS, GCP 또는 Azure)를 추상화합니다. 다음 섹션에 설명된 정보는 Amazon S3, Google Cloud Storage 및 Azure Blob Storage에 균일하게 적용됩니다.

{{% alert %}}
W&B Artifacts는 MinIO를 포함한 모든 Amazon S3 호환 인터페이스를 지원합니다. `AWS_S3_ENDPOINT_URL` 환경 변수를 MinIO 서버를 가리키도록 설정하면 아래 스크립트가 그대로 작동합니다.
{{% /alert %}}

다음과 같은 구조의 버킷이 있다고 가정합니다.

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` 아래에 이미지 모음인 데이터셋이 있습니다. 아티팩트로 추적해 보겠습니다.

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
기본적으로 W&B는 오브젝트 접두사를 추가할 때 10,000개의 오브젝트 제한을 적용합니다. `add_reference` 호출에서 `max_objects=`를 지정하여 이 제한을 조정할 수 있습니다.
{{% /alert %}}

새 reference artifact 인 `mnist:latest`는 일반 아티팩트와 유사하게 보이고 작동합니다. 유일한 차이점은 아티팩트가 ETag, 크기 및 버전 ID (오브젝트 버전 관리가 버킷에서 활성화된 경우)와 같은 S3/GCS/Azure 오브젝트에 대한 메타데이터로만 구성된다는 것입니다.

W&B는 기본 메커니즘을 사용하여 사용하는 클라우드 공급자를 기반으로 자격 증명을 찾습니다. 사용된 자격 증명에 대한 자세한 내용은 클라우드 공급자의 문서를 참조하십시오.

| 클라우드 공급자 | 자격 증명 문서 |
| -------------- | ------------------------- |
| AWS | [Boto3 문서](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP | [Google Cloud 문서](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure | [Azure 문서](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS의 경우 버킷이 구성된 사용자의 기본 리전에 있지 않으면 `AWS_REGION` 환경 변수를 버킷 리전과 일치하도록 설정해야 합니다.

일반 아티팩트와 유사하게 이 아티팩트와 상호 작용합니다. App UI에서 파일 브라우저를 사용하여 reference artifact 의 내용을 살펴보고 전체 종속성 그래프를 탐색하고 아티팩트의 버전 관리된 기록을 스캔할 수 있습니다.

{{% alert color="secondary" %}}
이미지, 오디오, 비디오 및 포인트 클라우드와 같은 풍부한 미디어는 버킷의 CORS 구성에 따라 App UI에서 렌더링되지 않을 수 있습니다. 버킷의 CORS 설정에서 **app.wandb.ai** 목록을 허용하면 App UI가 이러한 풍부한 미디어를 적절하게 렌더링할 수 있습니다.

개인 버킷의 경우 패널이 App UI에서 렌더링되지 않을 수 있습니다. 회사에 VPN이 있는 경우 VPN 내에서 IP를 허용하도록 버킷의 엑세스 정책을 업데이트할 수 있습니다.
{{% /alert %}}

### reference artifact 다운로드

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B는 아티팩트가 기록될 때 기록된 메타데이터를 사용하여 reference artifact 를 다운로드할 때 기본 버킷에서 파일을 검색합니다. 버킷에서 오브젝트 버전 관리를 활성화한 경우 W&B는 아티팩트가 기록될 당시의 파일 상태에 해당하는 오브젝트 버전을 검색합니다. 즉, 버킷 내용을 발전시키더라도 아티팩트가 트레이닝 당시 버킷의 스냅샷 역할을 하므로 지정된 모델이 트레이닝된 데이터의 정확한 반복을 계속 가리킬 수 있습니다.

{{% alert %}}
워크플로우의 일부로 파일을 덮어쓰는 경우 스토리지 버킷에서 '오브젝트 버전 관리'를 활성화하는 것이 좋습니다. 버킷에서 버전 관리를 활성화하면 덮어쓴 파일에 대한 참조가 있는 아티팩트가 이전 오브젝트 버전이 유지되므로 여전히 손상되지 않습니다.

유스 케이스에 따라 오브젝트 버전 관리를 활성화하는 방법에 대한 지침을 읽으십시오. [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/azure/storage/blobs/versioning-enable).
{{% /alert %}}

### 함께 묶기

다음 코드 예제는 Amazon S3, GCS 또는 Azure에서 트레이닝 작업에 제공되는 데이터셋을 추적하는 데 사용할 수 있는 간단한 워크플로우를 보여줍니다.

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# 아티팩트를 추적하고
# 이 run에 대한 입력으로 표시합니다. 새 아티팩트 버전은
# 버킷의 파일이 변경된 경우에만 기록됩니다.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# 여기서 트레이닝을 수행합니다...
```

모델을 추적하기 위해 트레이닝 스크립트가 모델 파일을 버킷에 업로드한 후 모델 아티팩트를 기록할 수 있습니다.

```python
import boto3
import wandb

run = wandb.init()

# 여기서 트레이닝을 수행합니다...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

{{% alert %}}
GCP 또는 Azure에 대한 참조로 아티팩트를 추적하는 방법에 대한 엔드투엔드 연습은 다음 리포트를 참조하십시오.

* [참조로 아티팩트 추적 가이드](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Microsoft Azure에서 참조 아티팩트 작업](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

### 파일 시스템 참조

데이터셋에 빠르게 엑세스하기 위한 또 다른 일반적인 패턴은 트레이닝 작업을 실행하는 모든 머신에서 원격 파일 시스템에 대한 NFS 마운트 지점을 노출하는 것입니다. 트레이닝 스크립트의 관점에서 파일이 로컬 파일 시스템에 있는 것처럼 보이기 때문에 클라우드 스토리지 버킷보다 훨씬 더 간단한 솔루션이 될 수 있습니다. 다행히 이러한 사용 편의성은 파일 시스템에 대한 참조를 추적하기 위해 Artifacts를 사용하는 데까지 확장됩니다 (마운트 여부와 관계없이).

다음과 같은 구조로 `/mount`에 파일 시스템이 마운트되어 있다고 가정합니다.

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` 아래에 이미지 모음인 데이터셋이 있습니다. 아티팩트로 추적해 보겠습니다.

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

기본적으로 W&B는 디렉토리에 대한 참조를 추가할 때 10,000개의 파일 제한을 적용합니다. `add_reference` 호출에서 `max_objects=`를 지정하여 이 제한을 조정할 수 있습니다.

URL에서 슬래시가 세 개 있다는 점에 유의하십시오. 첫 번째 구성 요소는 파일 시스템 참조 사용을 나타내는 `file://` 접두사입니다. 두 번째는 데이터셋 경로인 `/mount/datasets/mnist/`입니다.

결과 아티팩트인 `mnist:latest`는 일반 아티팩트와 마찬가지로 보이고 작동합니다. 유일한 차이점은 아티팩트가 크기 및 MD5 체크섬과 같은 파일에 대한 메타데이터로만 구성된다는 것입니다. 파일 자체는 시스템을 벗어나지 않습니다.

일반 아티팩트와 마찬가지로 이 아티팩트와 상호 작용할 수 있습니다. UI에서 파일 브라우저를 사용하여 reference artifact 의 내용을 찾아보고 전체 종속성 그래프를 탐색하고 아티팩트의 버전 관리된 기록을 스캔할 수 있습니다. 그러나 데이터 자체가 아티팩트에 포함되어 있지 않으므로 UI는 이미지, 오디오 등과 같은 풍부한 미디어를 렌더링할 수 없습니다.

reference artifact 를 다운로드하는 것은 간단합니다.

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

파일 시스템 참조의 경우 `download()` 작업은 참조된 경로에서 파일을 복사하여 아티팩트 디렉토리를 구성합니다. 위의 예에서 `/mount/datasets/mnist`의 내용은 `artifacts/mnist:v0/` 디렉토리에 복사됩니다. 아티팩트에 덮어쓴 파일에 대한 참조가 포함되어 있는 경우 아티팩트를 더 이상 재구성할 수 없으므로 `download()`에서 오류가 발생합니다.

모든 것을 함께 놓으면 다음은 마운트된 파일 시스템에서 트레이닝 작업에 제공되는 데이터셋을 추적하는 데 사용할 수 있는 간단한 워크플로우입니다.

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# 아티팩트를 추적하고
# 이 run에 대한 입력으로 표시합니다. 새 아티팩트 버전은
# 디렉토리 아래의 파일이
# 변경되었습니다.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# 여기서 트레이닝을 수행합니다...
```

모델을 추적하기 위해 트레이닝 스크립트가 모델 파일을 마운트 지점에 쓴 후 모델 아티팩트를 기록할 수 있습니다.

```python
import wandb

run = wandb.init()

# 여기서 트레이닝을 수행합니다...

# 디스크에 모델 쓰기

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```