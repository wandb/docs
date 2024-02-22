---
description: Track files saved outside the W&B such as in an Amazon S3 bucket, GCS
  bucket, HTTP file server, or even an NFS share.
displayed_sidebar: default
---

# 외부 파일 추적하기

<head>
	<title>참조 아티팩트를 사용한 외부 파일 추적</title>
</head>

예를 들어 Amazon S3 버킷, GCS 버킷, Azure blob, HTTP 파일 서버 또는 NFS 공유와 같은 W&B 시스템 외부에 저장된 파일을 추적하기 위해 **참조 아티팩트**를 사용하세요. W&B [CLI](https://docs.wandb.ai/ref/cli)를 사용하여 [W&B 실행](https://docs.wandb.ai/ref/python/run) 외부에서 아티팩트를 로그합니다.

### 실행 외부에서 아티팩트 로그하기

실행 외부에서 아티팩트를 로그할 때 W&B는 실행을 생성합니다. 각 아티팩트는 프로젝트에 속한 실행에 속하며; 아티팩트(버전)는 또한 컬렉션에 속하며, 타입이 있습니다.

W&B 실행 외부에서 W&B 서버로 아티팩트를 업로드하려면 [`wandb artifact put`](https://docs.wandb.ai/ref/cli/wandb-artifact/wandb-artifact-put) 명령을 사용하세요. 아티팩트가 속할 프로젝트 이름과 함께 아티팩트의 이름(`project/artifact_name`)을 제공하세요. 선택적으로 타입(`TYPE`)을 제공합니다. 아래 코드 조각에서 `PATH`를 업로드하려는 아티팩트의 파일 경로로 대체하세요.

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

지정한 프로젝트가 없으면 W&B는 새 프로젝트를 생성합니다. 아티팩트를 다운로드하는 방법에 대한 정보는 [아티팩트 다운로드 및 사용](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact)을 참조하세요.

## W&B 외부에서 아티팩트 추적하기

데이터세트 버전 관리 및 모델 계보를 위해 W&B 아티팩트를 사용하고, W&B 서버 외부에 저장된 파일을 추적하기 위해 **참조 아티팩트**를 사용하세요. 이 모드에서 아티팩트는 파일에 대한 메타데이터만 저장합니다. 예를 들어 URL, 크기, 체크섬 등입니다. 기본 데이터는 시스템을 떠나지 않습니다. 파일과 디렉터리를 W&B 서버에 저장하는 방법에 대한 정보는 [빠른 시작](https://docs.wandb.ai/guides/artifacts/artifacts-walkthrough)을 참조하세요.

참조 아티팩트를 구성하는 방법과 워크플로에 최적으로 통합하는 방법을 다음과 같이 설명합니다.

### Amazon S3 / GCS / Azure Blob Storage 참조

클라우드 스토리지 버킷의 참조를 추적하여 데이터세트 및 모델 버전 관리를 위해 W&B 아티팩트를 사용하세요. 아티팩트 참조를 사용하면 기존 저장소 레이아웃에 대한 수정 없이 버킷 위에 추적을 원활하게 더할 수 있습니다.

아티팩트는 기본 클라우드 스토리지 공급자(AWS, GCP 또는 Azure와 같은)를 추상화합니다. 다음 섹션에서 설명하는 정보는 Amazon S3, Google Cloud Storage 및 Azure Blob Storage에 동일하게 적용됩니다.

:::info
W&B 아티팩트는 MinIO를 포함한 모든 Amazon S3 호환 인터페이스를 지원합니다! AWS\_S3\_ENDPOINT\_URL 환경 변수를 MinIO 서버를 가리키도록 설정하면 아래 스크립트가 그대로 작동합니다.
:::

다음 구조를 가진 버킷이 있다고 가정합니다:

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` 아래에는 이미지 모음인 데이터세트가 있습니다. 아티팩트로 추적해 봅시다:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
:::caution
기본적으로 W&B는 객체 접두사를 추가할 때 10,000 개체 제한을 부과합니다. `add_reference` 호출 시 `max_objects=`를 지정하여 이 제한을 조정할 수 있습니다.
:::

우리의 새 참조 아티팩트 `mnist:latest`는 일반적인 아티팩트와 유사하게 보이고 동작합니다. 유일한 차이점은 아티팩트가 S3/GCS/Azure 객체에 대한 메타데이터만으로 구성된다는 것입니다, 예를 들어 ETag, 크기 및 버전 ID(버킷에서 객체 버전 관리가 활성화된 경우).

W&B는 클라우드 공급자가 사용하는 기본 메커니즘을 사용하여 자격 증명을 찾습니다. 사용하는 클라우드 공급자의 문서를 읽어 자격 증명에 대해 자세히 알아보세요:

| 클라우드 공급자 | 자격 증명 문서 |
| -------------- | ------------------------- |
| AWS            | [Boto3 문서](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud 문서](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure 문서](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

이 아티팩트와 일반 아티팩트와 유사하게 상호작용할 수 있습니다. 앱 UI에서 파일 브라우저를 사용하여 참조 아티팩트의 내용을 탐색하고, 전체 종속성 그래프를 탐색하며, 아티팩트의 버전 기록을 검토할 수 있습니다.

:::caution
이미지, 오디오, 비디오 및 포인트 클라우드와 같은 풍부한 미디어는 버킷의 CORS 설정에 따라 앱 UI에서 렌더링되지 않을 수 있습니다. **app.wandb.ai**를 버킷의 CORS 설정에 허용 목록에 추가하면 앱 UI가 해당 미디어를 올바르게 렌더링할 수 있습니다.

개인 버킷의 경우 패널이 앱 UI에서 렌더링되지 않을 수 있습니다. 회사에 VPN이 있는 경우 VPN 내의 IP를 허용 목록에 추가하여 버킷의 엑세스 정책을 업데이트할 수 있습니다.
:::

### 참조 아티팩트 다운로드하기

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B는 아티팩트가 로그될 때 기록된 메타데이터를 사용하여 참조 아티팩트를 다운로드할 때 기본 버킷에서 파일을 검색합니다. 버킷에 객체 버전 관리가 활성화된 경우, W&B는 아티팩트가 로그된 시점의 파일 상태에 해당하는 객체 버전을 검색합니다. 이는 버킷의 내용이 진화함에 따라 여전히 특정 모델이 학습된 데이터의 정확한 반복을 가리킬 수 있음을 의미합니다. 아티팩트는 학습 시점의 버킷 스냅샷 역할을 합니다.

:::info
워크플로의 일부로 파일을 덮어쓰는 경우 저장 버킷에 '객체 버전 관리'를 활성화하는 것이 좋습니다. 버전 관리가 버킷에 활성화되면 덮어쓰여진 파일에 대한 참조가 있는 아티팩트도 여전히 온전히 유지됩니다. 이전 객체 버전이 유지되기 때문입니다.

사용 사례에 따라 객체 버전 관리를 활성화하는 방법을 읽어보세요: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-enable).
:::

### 모든 것을 함께 묶기

다음 코드 예제는 Amazon S3, GCS 또는 Azure에서 학습 작업으로 피드되는 데이터세트를 추적하는 간단한 워크플로를 보여줍니다:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# 이 실행의 입력으로 아티팩트를 추적하고
# 한 번에 표시합니다. 버킷의 파일이 변경된 경우에만
# 새 아티팩트 버전이 기록됩니다.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# 여기서 학습을 수행하세요...
```

모델을 추적하려면 학습 스크립트가 모델 파일을 버킷에 업로드한 후 모델 아티팩트를 로그할 수 있습니다:

```python
import boto3
import wandb

run = wandb.init()

# 여기서 학습을 수행하세요...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

:::info
GCP 또는 Azure에 대한 참조로 아티팩트를 추적하는 방법에 대한 엔드 투 엔드 연습을 위해 다음 리포트를 읽어보세요:

* [참조로 아티팩트 추적 가이드](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Microsoft Azure에서 참조 아티팩트 작업하기](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
:::

### 파일 시스템 참조

데이터세트에 빠르게 엑세스하는 또 다른 일반적인 패턴은 모든 학습 작업을 실행하는 기계에 원격 파일 시스템에 대한 NFS 마운트 포인트를 노출하는 것입니다. 이는 클라우드 스토리지 버킷보다 더 간단한 솔루션이 될 수 있습니다. 학습 스크립트의 관점에서 파일이 로컬 파일 시스템에 있는 것처럼 보입니다. 다행히도, 파일 시스템 참조를 추적하는 데 있어 Artifacts 사용의 용이성이 그대로 유지됩니다 — 마운트된 것이든 그렇지 않든.

다음 구조를 가진 파일 시스템이 `/mount`에 마운트되어 있다고 가정합니다:

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` 아래에는 이미지 모음인 데이터세트가 있습니다. 아티팩트로 추적해 봅시다:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

기본적으로 W&B는 디렉터리에 대한 참조를 추가할 때 10,000 파일 제한을 부과합니다. `add_reference` 호출 시 `max_objects=`를 지정하여 이 제한을 조정할 수 있습니다.

URL에서 세 번째 슬래시를 주목하세요. 첫 번째 구성 요소는 파일 시스템 참조의 사용을 나타내는 `file://` 접두사입니다. 두 번째는 데이터세트 경로, `/mount/datasets/mnist/`입니다.

결과적으로 생성된 아티팩트 `mnist:latest`는 일반적인 아티팩트와 같이 보이고 작동합니다. 유일한 차이점은 아티팩트가 파일에 대한 메타데이터만으로 구성된다는 것입니다, 예를 들어 파일 크기와 MD5 체크섬 등입니다. 파일 자체는 시스템을 떠나지 않습니다.

이 아티팩트와 일반 아티팩트와 유사하게 상호작용할 수 있습니다. UI에서 파일 브라우저를 사용하여 참조 아티팩트의 내용을 둘러보고, 전체 종속성 그래프를 탐색하며, 아티팩트의 버전 기록을 검토할 수 있습니다. 그러나 UI는 데이터 자체가 아티팩트 내에 포함되어 있지 않기 때문에 이미지, 오디오 등의 풍부한 미디어를 렌더링할 수 없습니다.

참조 아티팩트를 다운로드하는 것은 간단합니다:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

파일 시스템 참조의 경우, `download()` 작업은 아티팩트 디렉터리를 구성하기 위해 참조된 경로에서 파일을 복사합니다. 위 예제에서 `/mount/datasets/mnist`의 내용은 `artifacts/mnist:v0/` 디렉터리에 복사됩니다. 아티팩트가 덮어쓰여진 파일에 대한 참조를 포함하는 경우, `download()`는 아티팩트를 더 이상 재구성할 수 없으므로 오류를 발생시킵니다.

모든 것을 함께 묶어, 마운트된 파일 시스템 아래에 있는 데이터세트를 추적하여 학습 작업으로 피드하는 간단한 워크플로를 사용할 수 있습니다:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# 이 실행의 입력으로 아티팩트를 추적하고
# 한 번에 표시합니다. 디렉터리 아래의 파일이 변경된 경우에만
# 새 아