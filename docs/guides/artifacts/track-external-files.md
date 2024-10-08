---
title: Track external files
description: W&B 외부에 저장된 파일을 Amazon S3 버킷, GCS 버킷, HTTP 파일 서버, 또는 NFS 공유에서 트래킹합니다.
displayed_sidebar: default
---

**레퍼런스 Artifacts**를 사용하여 W&B 시스템 외부에 저장된 파일을 추적하세요. 예를 들어 Amazon S3 버킷, GCS 버킷, Azure 블롭, HTTP 파일 서버 또는 NFS 공유 등에 저장된 파일입니다. [W&B Run](/ref/python/run) 외부에서 W&B [CLI](/ref/cli)를 사용하여 아티팩트를 로그하세요.

### Runs 외부에서 artifacts 로그하기

W&B는 run 외부에서 artifact를 로그할 때 run을 생성합니다. 각 artifact는 run에 속하며, run은 프로젝트에 속합니다; artifact(버전)는 또한 컬렉션에 속하며, 유형을 가집니다.

W&B run 외부에서 아티팩트를 W&B 서버에 업로드하려면 [`wandb artifact put`](/ref/cli/wandb-artifact/wandb-artifact-put) 코맨드를 사용하세요. 아티팩트가 속할 프로젝트의 이름과 아티팩트의 이름(`project/artifact_name`)을 제공하세요. 옵션으로 유형(`TYPE`)을 지정할 수 있습니다. 아래 코드 조각의 `PATH`를 업로드하고자 하는 아티팩트의 파일 경로로 대체하세요.

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

지정한 프로젝트가 존재하지 않으면 W&B는 새로운 프로젝트를 생성합니다. 아티팩트를 다운로드하는 방법에 대한 정보는 [Download and use artifacts](/guides/artifacts/download-and-use-an-artifact)를 참조하세요.

## W&B 외부에서 artifacts 추적하기

W&B Artifacts를 데이터셋 버전 관리 및 모델 이력 추적에 사용하고, **레퍼런스 Artifacts**를 사용하여 W&B 서버 외부에 저장된 파일을 추적하세요. 이 모드에서 아티팩트는 파일에 대한 메타데이터(예: URL, 크기, 체크섬)만 저장합니다. 기본 데이터는 시스템을 절대 떠나지 않습니다. W&B 서버에 파일과 디렉토리를 저장하는 방법에 대한 정보는 [Quick start](/guides/artifacts/artifacts-walkthrough)를 참조하세요.

다음은 레퍼런스 Artifacts를 구성하는 방법과 이를 워크플로우에 최적으로 통합하는 방법을 설명합니다.

### Amazon S3 / GCS / Azure Blob Storage References

클라우드 스토리지 버킷의 참조를 추적하기 위해 데이터셋과 모델 버전 관리에 W&B Artifacts를 사용하세요. Artifact 참조를 통해 기존 스토리지 레이아웃을 변경하지 않고도 버킷에 추적 레이어를 매끄럽게 추가할 수 있습니다.

Artifacts는 하위 클라우드 스토리지 벤더(AWS, GCP 또는 Azure)를 추상화합니다. 다음 섹션에 설명된 정보는 Amazon S3, Google Cloud Storage 및 Azure Blob Storage에 동일하게 적용됩니다.

:::info
W&B Artifacts는 MinIO를 포함하여 모든 Amazon S3 호환 인터페이스를 지원합니다! AWS_S3_ENDPOINT_URL 환경 변수를 MinIO 서버로 설정하면 아래 스크립트가 그대로 작동합니다.
:::

다음 구조를 가진 버킷을 가지고 있다고 가정합니다:

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` 아래에는 이미지 모음인 데이터셋이 있습니다. 이를 아티팩트로 추적해 봅시다:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
:::caution
기본적으로, W&B는 객체 접두사를 추가할 때 10,000 객체 제한을 두고 있습니다. 이 제한은 `add_reference` 호출 시 `max_objects=`를 지정하여 조정할 수 있습니다.
:::

새로운 레퍼런스 아티팩트 `mnist:latest`는 일반 아티팩트처럼 보이고 작동합니다. 유일한 차이점은 아티팩트가 S3/GCS/Azure 오브젝트에 대한 메타데이터(예: ETag, 크기 및 버전 ID (버킷에서 객체 버전 관리가 활성화된 경우)만으로 구성된다는 점입니다.

W&B는 사용 중인 클라우드 제공 업체에 따라 자격 증명을 찾기 위한 기본 메커니즘을 사용합니다. 사용하려는 자격 증명에 대해 알아보려면 클라우드 제공 업체의 설명서를 참조하세요:

| Cloud provider | Credentials Documentation |
| -------------- | ------------------------- |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS의 경우, 버킷이 설정된 사용자의 기본 지역에 있지 않으면 `AWS_REGION` 환경 변수를 버킷의 지역과 일치하도록 설정해야 합니다.

이 아티팩트와 상호 작용하는 것은 일반 아티팩트와 유사합니다. 앱 UI에서 파일 브라우저를 사용하여 아티팩트의 내용을 탐색하고, 전체 의존성 그래프를 탐색하며, 아티팩트의 버전 기록을 훑어볼 수 있습니다.

:::caution
이미지, 오디오, 비디오 및 포인트 클라우드와 같은 리치 미디어는 버킷의 CORS 설정에 따라 앱 UI에서 렌더링되지 않을 수 있습니다. 버킷의 CORS 설정에서 **app.wandb.ai**를 허용 목록에 추가하면 앱 UI가 이러한 리치 미디어를 적절하게 렌더링할 수 있습니다.

개인 버킷의 경우 패널이 앱 UI에서 렌더링되지 않을 수 있습니다. 회사에 VPN이 있는 경우, 버킷의 엑세스 정책을 업데이트하여 VPN 내의 IP를 허용 목록에 추가할 수 있습니다.
:::

### 레퍼런스 artifact 다운로드하기

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B는 레퍼런스 아티팩트를 다운로드할 때 아티팩트가 로그될 당시 기록된 메타데이터를 사용하여 기본 버킷에서 파일을 가져옵니다. 버킷에 객체 버전 관리가 활성화되어 있는 경우, W&B는 아티팩트가 로그될 당시의 파일 상태와 일치하는 객체 버전을 가져옵니다. 즉, 버킷의 내용이 변경되더라도, 아티팩트가 트레이닝 시점의 버킷 스냅샷으로 작동하므로 모델이 트레이닝된 데이터의 정확한 반복물을 여전히 가리킬 수 있습니다.

:::info
W&B는 워크플로우의 일부로 파일을 덮어쓰는 경우 스토리지 버킷에 '객체 버전 관리'를 활성화할 것을 권장합니다. 버킷에 버전 관리가 활성화된 경우, 덮어쓰여진 파일에 대한 참조를 포함하는 아티팩트는 여전히 완전한 상태로 남습니다. 예전 객체 버전이 유지되기 때문입니다.

사용 사례에 기반하여 객체 버전 관리를 활성화하기 위한 지침을 읽으세요: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-enable).
:::

### 모두 연결하기

다음 코드 예제는 트레이닝 작업에 들어가는 Amazon S3, GCS 또는 Azure의 데이터셋을 추적하는 데 사용할 수 있는 간단한 워크플로우를 보여줍니다:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# 아티팩트를 추적하고 이를
# 동시에 이 run의 입력으로 표시합니다. 새로운 아티팩트 버전은
# 버킷의 파일이 변경된 경우에만 로그됩니다.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# 여기에서 트레이닝 수행...
```

모델을 추적하기 위해, 트레이닝 스크립트가 모델 파일을 버킷에 업로드한 후 모델 아티팩트를 로그할 수 있습니다:

```python
import boto3
import wandb

run = wandb.init()

# 여기에서 트레이닝 수행...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

:::info
GCP 또는 Azure에 대한 참조를 통해 아티팩트를 추적하는 방법에 대한 엔드투엔드 워크스루를 보려면 다음 보고서를 읽어보세요:

* [Guide to Tracking Artifacts by Reference](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Working with Reference Artifacts in Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
:::

### 파일시스템 참조

데이터셋에 빠르게 엑세스하기 위한 또 다른 일반적인 패턴은 모든 트레이닝 작업을 실행하는 기계에 원격 파일 시스템에 대한 NFS 마운트 포인트를 노출하는 것입니다. 이는 클라우드 스토리지 버킷보다 더 단순한 솔루션일 수 있습니다. 트레이닝 스크립트의 관점에서 파일은 로컬 파일 시스템에 앉아 있는 것처럼 보이기 때문입니다. 다행히도, 그 사용의 용이성은 마운트되었든 아니든 파일 시스템에 대한 참조를 추적하기 위한 Artifacts 사용까지 확장됩니다.

다음 구조로 파일 시스템이 `/mount`에 마운트되어 있다고 가정합니다:

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` 아래에는 이미지 모음인 데이터셋이 있습니다. 이를 아티팩트로 추적해 봅시다:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

기본적으로 W&B는 디렉토리에 대한 참조를 추가할 때 10,000 파일 제한을 둡니다. 이 제한은 `add_reference` 호출 시 `max_objects=`를 지정하여 조정할 수 있습니다.

URL에서 세 개의 슬래시를 주의하세요. 첫 번째 구성요소는 파일 시스템 참조 사용을 나타내는 `file://` 접두사입니다. 두 번째 구성요소는 우리의 데이터셋 경로인 `/mount/datasets/mnist/`입니다.

결과적인 아티팩트 `mnist:latest`는 일반 아티팩트처럼 보이고 동작합니다. 유일한 차이점은 아티팩트가 파일의 크기 및 MD5 체크섬과 같은 메타데이터로만 구성된다는 점입니다. 파일 자체는 시스템을 떠나지 않습니다.

이 아티팩트와 상호 작용하는 것은 일반 아티팩트와 유사합니다. UI에서 파일 브라우저를 통해 참조 아티팩트의 내용을 탐색하고, 전체 의존성 그래프를 탐색하며, 아티팩트의 버전 기록을 검사할 수 있습니다. 그러나 UI는 이미지, 오디오 등과 같은 리치 미디어를 렌더링할 수 없으므로 데이터 자체가 아티팩트에 포함되어 있지 않습니다.

레퍼런스 아티팩트를 다운로드하는 것은 간단합니다:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

파일 시스템 참조의 경우, `download()` 작업은 아티팩트 디렉토리를 구성하기 위해 참조된 경로에서 파일을 복사합니다. 위의 예에서 `/mount/datasets/mnist`의 내용은 `artifacts/mnist:v0/` 디렉토리에 복사됩니다. 아티팩트가 덮어쓰여진 파일에 대한 참조를 포함하는 경우, `download()`는 아티팩트를 더 이상 재구성할 수 없으므로 오류를 발생시킵니다.

모든 것을 종합하면, 여기에 트레이닝 작업에 들어가는 마운트된 파일 시스템의 데이터셋을 추적하는 데 사용할 수 있는 간단한 워크플로우가 있습니다:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# 아티팩트를 추적하고 이를
# 동시에 이 run의 입력으로 표시합니다.
# 새로운 아티팩트 버전은 디렉토리 아래의 파일이
# 변경된 경우에만 로그됩니다.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# 여기에서 트레이닝 수행...
```

모델을 추적하기 위해, 트레이닝 스크립트가 마운트 지점에 모델 파일을 기록한 후 모델 아티팩트를 로그할 수 있습니다:

```python
import wandb

run = wandb.init()

# 여기에서 트레이닝 수행...

# 디스크에 모델을 기록

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```