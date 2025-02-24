---
title: Delete an artifact
description: App UI를 사용하여 대화형으로 또는 W&B SDK를 사용하여 프로그래밍 방식으로 Artifacts를 삭제합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-manage-data-delete-artifacts
    parent: manage-data
---

Artifacts를 App UI를 통해 상호 작용적으로 삭제하거나 W&B SDK를 통해 프로그래밍 방식으로 삭제할 수 있습니다. 아티팩트를 삭제하면 W&B는 해당 Artifact를 *소프트 삭제*로 표시합니다. 즉, 아티팩트는 삭제 대상으로 표시되지만 파일은 즉시 스토리지에서 삭제되지 않습니다.

아티팩트의 내용은 정기적으로 실행되는 가비지 컬렉션 프로세스가 삭제 대상으로 표시된 모든 아티팩트를 검토할 때까지 소프트 삭제 또는 삭제 보류 상태로 유지됩니다. 가비지 컬렉션 프로세스는 아티팩트 및 관련 파일이 이전 또는 이후 아티팩트 버전에 사용되지 않은 경우 스토리지에서 연결된 파일을 삭제합니다.

이 페이지의 섹션에서는 특정 아티팩트 버전을 삭제하는 방법, 아티팩트 컬렉션을 삭제하는 방법, 에일리어스가 있거나 없는 아티팩트를 삭제하는 방법 등을 설명합니다. TTL 정책을 사용하여 W&B에서 아티팩트가 삭제되는 시기를 예약할 수 있습니다. 자세한 내용은 [Artifact TTL 정책으로 데이터 보존 관리]({{< relref path="./ttl.md" lang="ko" >}})을 참조하세요.

{{% alert %}}
TTL 정책으로 삭제가 예약된 Artifact, W&B SDK로 삭제된 Artifact 또는 W&B App UI로 삭제된 Artifact는 먼저 소프트 삭제됩니다. 소프트 삭제된 Artifact는 하드 삭제되기 전에 가비지 컬렉션을 거칩니다.
{{% /alert %}}

### 아티팩트 버전 삭제

아티팩트 버전을 삭제하려면:

1. 아티팩트 이름을 선택합니다. 이렇게 하면 아티팩트 보기가 확장되고 해당 아티팩트와 연결된 모든 아티팩트 버전이 나열됩니다.
2. 아티팩트 목록에서 삭제하려는 아티팩트 버전을 선택합니다.
3. 워크스페이스 오른쪽에서 케밥 드롭다운을 선택합니다.
4. 삭제를 선택합니다.

아티팩트 버전은 [delete()]({{< relref path="/ref/python/artifact#delete" lang="ko" >}}) 메소드를 통해 프로그래밍 방식으로 삭제할 수도 있습니다. 아래 예제를 참조하십시오.

### 에일리어스가 있는 여러 아티팩트 버전 삭제

다음 코드 예제에서는 에일리어스가 연결된 Artifact를 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 엔티티, 프로젝트 이름 및 run ID를 제공합니다.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

아티팩트에 에일리어스가 하나 이상 있는 경우 `delete_aliases` 파라미터를 부울 값인 `True`로 설정하여 에일리어스를 삭제합니다.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # 하나 이상의 에일리어스가 있는 Artifact를 삭제하려면
    # delete_aliases=True를 설정합니다.
    artifact.delete(delete_aliases=True)
```

### 특정 에일리어스가 있는 여러 아티팩트 버전 삭제

다음 코드는 특정 에일리어스가 있는 여러 아티팩트 버전을 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 엔티티, 프로젝트 이름 및 run ID를 제공합니다. 삭제 로직을 직접 작성하십시오.

```python
import wandb

runs = api.run("entity/project_name/run_id")

# 'v3' 및 'v4' 에일리어스가 있는 아티팩트 삭제
for artifact_version in runs.logged_artifacts():
    # 삭제 로직을 직접 작성하십시오.
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### 에일리어스가 없는 아티팩트의 모든 버전 삭제

다음 코드 조각은 에일리어스가 없는 아티팩트의 모든 버전을 삭제하는 방법을 보여줍니다. `wandb.Api`에서 `project` 및 `entity` 키에 대한 프로젝트 및 엔티티 이름을 각각 제공합니다. `<>`를 Artifact 이름으로 바꿉니다.

```python
import wandb

# wandb.Api 메소드를 사용할 때 엔티티 및 프로젝트 이름을 제공합니다.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # 유형 및 이름 제공
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest'와 같은 에일리어스가 없는 버전을 정리합니다.
    # 참고: 원하는 삭제 로직을 여기에 넣을 수 있습니다.
    if len(v.aliases) == 0:
        v.delete()
```

### 아티팩트 컬렉션 삭제

아티팩트 컬렉션을 삭제하려면:

1. 삭제하려는 아티팩트 컬렉션으로 이동하여 마우스를 올립니다.
2. 아티팩트 컬렉션 이름 옆에 있는 케밥 드롭다운을 선택합니다.
3. 삭제를 선택합니다.

[delete()]({{< relref path="/ref/python/artifact.md#delete" lang="ko" >}}) 메소드를 사용하여 프로그래밍 방식으로 아티팩트 컬렉션을 삭제할 수도 있습니다. `wandb.Api`에서 `project` 및 `entity` 키에 대한 프로젝트 및 엔티티 이름을 각각 제공합니다.

```python
import wandb

# wandb.Api 메소드를 사용할 때 엔티티 및 프로젝트 이름을 제공합니다.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&B 호스팅 방식에 따라 가비지 컬렉션을 활성화하는 방법
W&B의 공유 클라우드를 사용하는 경우 가비지 컬렉션이 기본적으로 활성화됩니다. W&B를 호스팅하는 방식에 따라 가비지 컬렉션을 활성화하기 위해 추가 단계를 수행해야 할 수 있습니다.

* `GORILLA_ARTIFACT_GC_ENABLED` 환경 변수를 true로 설정합니다. `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/object-versioning) 또는 [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)와 같은 다른 스토리지 공급자를 사용하는 경우 버킷 버전을 활성화합니다. Azure를 사용하는 경우 [소프트 삭제를 활성화](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)합니다.
  {{% alert %}}
  Azure의 소프트 삭제는 다른 스토리지 공급자의 버킷 버전 관리와 같습니다.
  {{% /alert %}}

다음 표에서는 배포 유형에 따라 가비지 컬렉션을 활성화하기 위한 요구 사항을 충족하는 방법을 설명합니다.

`X`는 요구 사항을 충족해야 함을 나타냅니다.

|                                                | 환경 변수    | 버전 관리 활성화 |
| -----------------------------------------------| ------------------------| ----------------- |
| 공유 클라우드                                   |                         |                   |
| [보안 스토리지 커넥터]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})가 있는 공유 클라우드|                         | X                 |
| 전용 클라우드                                |                         |                   |
| [보안 스토리지 커넥터]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})가 있는 전용 클라우드|                         | X                 |
| 고객 관리 클라우드                         | X                       | X                 |
| 고객 관리 온프레미스                       | X                       | X                 |

{{% alert %}}note
보안 스토리지 커넥터는 현재 Google Cloud Platform 및 Amazon Web Services에서만 사용할 수 있습니다.
{{% /alert %}}
