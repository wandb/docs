---
title: Delete an artifact
description: App UI를 사용하여 상호 작용 방식으로 또는 W&B SDK를 사용하여 프로그래밍 방식으로 아티팩트를 삭제합니다.
displayed_sidebar: default
---

아티팩트는 App UI에서 대화형으로 삭제하거나 W&B SDK를 사용하여 프로그래매틱하게 삭제할 수 있습니다. 아티팩트를 삭제하면, W&B는 해당 아티팩트를 *소프트 삭제*로 표시합니다. 즉, 아티팩트는 삭제 대기로 표시되지만 파일은 즉시 스토리지에서 삭제되지 않습니다.

아티팩트의 내용은 소프트 삭제 또는 삭제 대기 상태로 남아 있으며, 정기적으로 수행되는 가비지 수집 프로세스가 삭제 대기로 표시된 모든 아티팩트를 검토할 때까지 유지됩니다. 가비지 수집 프로세스는 이전 또는 이후 아티팩트 버전에서 사용되지 않은 경우에만 관련된 파일을 스토리지에서 삭제합니다.

이 페이지의 섹션에서는 특정 아티팩트 버전을 삭제하는 방법, 아티팩트 컬렉션을 삭제하는 방법, 에일리어스가 있는 아티팩트를 삭제하는 방법 등을 설명합니다. TTL 정책을 사용하여 W&B에서 아티팩트가 삭제되는 시점을 예약할 수 있습니다. 자세한 내용은 [Artifact TTL 정책을 사용한 데이터 보존 관리](./ttl.md)를 참조하세요.

:::note
TTL 정책에 따라 삭제 스케줄이 지정된 아티팩트, W&B SDK를 통해 삭제된 아티팩트, 또는 W&B App UI에서 삭제된 아티팩트는 먼저 소프트 삭제됩니다. 소프트 삭제된 아티팩트는 하드 삭제되기 전에 가비지 수집을 거치게 됩니다.
:::

### 아티팩트 버전 삭제

아티팩트 버전을 삭제하려면:

1. 아티팩트의 이름을 선택합니다. 그러면 아티팩트 뷰가 확장되고 해당 아티팩트와 관련된 모든 아티팩트 버전이 나열됩니다.
2. 아티팩트 목록에서 삭제하려는 아티팩트 버전을 선택합니다.
3. 워크스페이스의 오른쪽에 있는 케밥 드롭다운을 선택합니다.
4. Delete를 선택합니다.

아티팩트 버전은 또한 [delete()](/ref/python/artifact#delete) 메소드를 통해 프로그래매틱하게 삭제할 수 있습니다. 아래 예제를 참고하세요.

### 에일리어스와 함께 여러 아티팩트 버전 삭제

다음 코드 예제는 에일리어스를 가진 아티팩트를 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 entity, 프로젝트 이름, run ID를 제공하세요.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

아티팩트에 하나 이상의 에일리어스가 있는 경우, 에일리어스를 삭제하려면 `delete_aliases` 파라미터를 `True`로 설정하세요.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # delete_aliases=True로 설정하여
    # 하나 이상의 에일리어스를 가진
    # 아티팩트를 삭제합니다
    artifact.delete(delete_aliases=True)
```

### 특정 에일리어스를 가진 여러 아티팩트 버전 삭제

다음 코드는 특정 에일리어스를 가진 여러 아티팩트 버전을 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 entity, 프로젝트 이름, run ID를 제공하세요. 삭제 로직은 자신만의 것으로 바꾸세요:

```python
import wandb

runs = api.run("entity/project_name/run_id")

# 'v3'와 'v4' 에일리어스를 가진 아티팩트 삭제
for artifact_version in runs.logged_artifacts():
    # 자신만의 삭제 로직으로 대체하세요.
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### 에일리어스가 없는 아티팩트의 모든 버전 삭제

다음 코드조각은 에일리어스가 없는 아티팩트의 모든 버전을 삭제하는 방법을 보여줍니다. `project` 및 `entity` 키에 대한 프로젝트와 entity의 이름을 각각 `wandb.Api`에 제공하세요. `<>`를 자신의 아티팩트 이름으로 교체하세요:

```python
import wandb

# wandb.Api 메소드를 사용할 때 entity와 
# 프로젝트 이름을 제공하세요.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # 타입과 이름을 제공
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest'와 같은 에일리어스가 없는 버전을 정리합니다.
    # NOTE: 원하시는 삭제 로직을 여기에 추가할 수 있습니다.
    if len(v.aliases) == 0:
        v.delete()
```

### 아티팩트 컬렉션 삭제

아티팩트 컬렉션을 삭제하려면:

1. 삭제하려는 아티팩트 컬렉션으로 이동하고 해당 컬렉션 위로 마우스를 올립니다.
3. 아티팩트 컬렉션 이름 옆에 있는 케밥 드롭다운을 선택합니다.
4. Delete를 선택합니다.

아티팩트 컬렉션은 [delete()](../../ref/python/artifact.md#delete) 메소드를 사용하여 프로그래매틱하게 삭제할 수도 있습니다. `project` 및 `entity` 키에 대한 프로젝트와 entity의 이름을 각각 제공하세요:

```python
import wandb

# wandb.Api 메소드를 사용할 때 entity와 
# 프로젝트 이름을 제공하세요.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&B 호스팅 방법에 따른 가비지 수집 활성화 방법
W&B의 공유 클라우드를 사용하는 경우 기본적으로 가비지 수집이 활성화됩니다. W&B를 호스팅하는 방법에 따라 가비지 수집을 활성화하기 위해 추가 단계가 필요할 수 있습니다. 여기에는 다음이 포함됩니다:

* `GORILLA_ARTIFACT_GC_ENABLED` 환경 변수를 true로 설정: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/object-versioning) 또는 [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)와 같은 스토리지 제공업체를 사용하는 경우 버킷 버전 관리를 활성화합니다. Azure를 사용하는 경우, [소프트 삭제를 활성화](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)하세요.
  :::note
  Azure의 소프트 삭제는 다른 스토리지 제공업체의 버킷 버전 관리와 동등합니다.
  :::

다음 표는 배포 유형에 따라 가비지 수집을 활성화하기 위한 요구 사항을 충족하는 방법을 설명합니다.

`X`는 요구 사항을 충족해야 함을 나타냅니다:

|                                                | 환경 변수   | 버전 관리 활성화 | 
| -----------------------------------------------| ------------| -----------------| 
| 공유 클라우드                                  |             |                   | 
| [보안 스토리지 커넥터](../hosting/data-security/secure-storage-connector.md)를 사용한 공유 클라우드|                         | X                 | 
| 전용 클라우드                                  |             |                   | 
| [보안 스토리지 커넥터](../hosting/data-security/secure-storage-connector.md)를 사용한 전용 클라우드|                         | X                 | 
| 고객 관리 클라우드                            | X           | X                 | 
| 고객 관리 온프레미스                           | X           | X                 |

:::note
보안 스토리지 커넥터는 현재 Google Cloud Platform 및 Amazon Web Services에서만 사용 가능합니다.
:::