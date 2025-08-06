---
title: 아티팩트 삭제
description: App UI를 사용하여 상호작용적으로 또는 W&B SDK를 통해 프로그래밍 방식으로 아티팩트를 삭제할 수 있습니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-manage-data-delete-artifacts
    parent: manage-data
---

App UI에서 인터랙티브하게 또는 W&B SDK로 프로그래밍 방식으로 Artifacts를 삭제할 수 있습니다. 아티팩트를 삭제하면 W&B는 해당 아티팩트를 *소프트 삭제* 상태로 표시합니다. 즉, 아티팩트가 삭제 대상으로 표시되지만 파일이 즉시 스토리지에서 삭제되지는 않습니다.

이 아티팩트의 내용은 소프트 삭제, 즉 삭제 대기 상태로 남아있으며, 정기적으로 실행되는 가비지 컬렉션 프로세스가 삭제 대상으로 표시된 모든 아티팩트를 확인합니다. 이 과정에서, 해당 아티팩트와 관련된 파일이 앞선 버전이나 이후 버전의 아티팩트에서 사용되지 않는 경우, 연관된 파일이 스토리지에서 실제로 삭제됩니다.

이 페이지에서는 특정 아티팩트 버전을 삭제하는 방법, 아티팩트 컬렉션 전체를 삭제하는 방법, 에일리어스가 있는 또는 없는 아티팩트를 삭제하는 방법 등 다양한 삭제 방법을 설명합니다. TTL 정책을 통해 Artifacts의 삭제 시점을 예약할 수 있습니다. 자세한 내용은 [아티팩트 TTL 정책으로 데이터 보존 관리하기]({{< relref path="./ttl.md" lang="ko" >}})를 참조하세요.

{{% alert %}}
TTL 정책으로 삭제 예약된 Artifacts, W&B SDK 또는 App UI에서 삭제된 Artifacts는 우선 소프트 삭제됩니다. 소프트 삭제된 Artifacts는 하드 삭제되기 전에 가비지 컬렉션을 거치게 됩니다.
{{% /alert %}}

### 아티팩트 버전 삭제하기

아티팩트 버전을 삭제하려면 아래 절차를 따르세요:

1. 삭제하려는 아티팩트의 이름을 선택하세요. 그러면 해당 아티팩트와 연관된 모든 버전이 표시됩니다.
2. Artifacts 목록에서 삭제할 아티팩트 버전을 선택하세요.
3. 워크스페이스 우측에 있는 케밥(더보기) 드롭다운 메뉴를 선택하세요.
4. 삭제(Delete)를 클릭하세요.

아티팩트 버전은 [delete()]({{< relref path="/ref/python/sdk/classes/artifact.md#delete" lang="ko" >}}) 메소드를 통해 프로그래밍 방식으로도 삭제할 수 있습니다. 아래 예시를 참고하세요.

### 에일리어스가 있는 여러 아티팩트 버전 삭제하기

다음 코드 예시는 에일리어스가 연결된 Artifacts를 삭제하는 방법을 보여줍니다. 해당 Artifacts를 생성한 엔티티, 프로젝트 이름, run ID를 입력하세요.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

`delete_aliases` 파라미터를 `True`로 설정하면, 아티팩트에 하나 이상의 에일리어스가 있을 경우 에일리어스도 함께 삭제됩니다.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # delete_aliases=True로 설정하면
    # 에일리어스가 있는 아티팩트도 함께 삭제됩니다
    artifact.delete(delete_aliases=True)
```

### 특정 에일리어스를 가진 여러 아티팩트 버전 삭제하기

다음 코드는 특정 에일리어스를 가진 여러 아티팩트 버전을 삭제하는 방법을 보여줍니다. 해당 Artifacts를 생성한 엔티티, 프로젝트 이름, run ID를 입력하세요. 삭제 로직은 필요에 따라 변경하세요:

```python
import wandb

runs = api.run("entity/project_name/run_id")

# 'v3'와 'v4' 에일리어스가 있는 아티팩트 버전 삭제
for artifact_version in runs.logged_artifacts():
    # 원하는 삭제 로직으로 변경 가능합니다.
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### 에일리어스가 없는 모든 아티팩트 버전 삭제하기

다음 코드조각은 에일리어스가 없는 모든 아티팩트 버전을 삭제하는 방법을 보여줍니다. `wandb.Api`의 `project`와 `entity` 키에 프로젝트명과 엔티티명을 입력하세요. `<>` 부분에는 본인의 아티팩트 이름을 넣으세요:

```python
import wandb

# wandb.Api 메소드 사용 시 엔티티와 프로젝트 이름을 입력합니다.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # 타입과 이름 입력
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest'와 같은 에일리어스가 없는 버전 정리
    # 원하는 삭제 로직을 적용하실 수 있습니다.
    if len(v.aliases) == 0:
        v.delete()
```

### 아티팩트 컬렉션 삭제하기

아티팩트 컬렉션을 삭제하려면:

1. 삭제하려는 아티팩트 컬렉션으로 이동해 마우스를 올리세요.
2. 아티팩트 컬렉션 이름 옆의 케밥 드롭다운을 선택하세요.
3. 삭제(Delete)를 클릭하세요.

아티팩트 컬렉션 역시 [delete()]({{< relref path="/ref/python/sdk/classes/artifact.md#delete" lang="ko" >}}) 메소드로 프로그래밍 방식으로도 삭제할 수 있습니다. `wandb.Api`의 `project`와 `entity` 키에 프로젝트명과 엔티티명을 각각 입력하세요:

```python
import wandb

# wandb.Api 메소드 사용 시 엔티티와 프로젝트 이름을 입력합니다.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&B의 호스팅 방식에 따른 가비지 컬렉션 활성화 방법
W&B 공유 클라우드를 사용할 경우 가비지 컬렉션은 기본적으로 활성화되어 있습니다. 호스팅 환경에 따라 추가 설정이 필요할 수 있습니다. 여기에는 다음이 포함됩니다:


* `GORILLA_ARTIFACT_GC_ENABLED` 환경 변수를 true로 설정: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/object-versioning) 또는 [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)와 같은 스토리지 제공업체를 사용할 경우 버킷 버전 관리를 활성화하세요. Azure를 사용할 경우 [소프트 삭제 활성화](https://learn.microsoft.com/azure/storage/blobs/soft-delete-blob-overview)가 필요합니다.
  {{% alert %}}
  Azure의 소프트 삭제 기능은 다른 스토리지 제공업체의 버킷 버전 관리와 동일합니다.
  {{% /alert %}}

아래 표는 배포 유형에 따라 가비지 컬렉션 활성화 요건을 충족하는 방법을 보여줍니다.

`X`가 표시된 항목은 해당 요건을 반드시 만족해야 함을 의미합니다:

|                                                | 환경 변수                | 버전 관리 활성화  | 
| -----------------------------------------------| ------------------------| ----------------- | 
| 공유 클라우드                                  |                         |                   | 
| 공유 클라우드에서 [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 사용 시|                         | X                 | 
| 전용 클라우드                                  |                         |                   | 
| 전용 클라우드에서 [Secure Storage Connector]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 사용 시|                         | X                 | 
| 고객이 관리하는 클라우드                       | X                       | X                 | 
| 고객이 관리하는 온프레미스                     | X                       | X                 | 
 

{{% alert %}}note
Secure Storage Connector는 현재 Google Cloud Platform과 Amazon Web Services에서만 지원됩니다.
{{% /alert %}}