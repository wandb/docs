---
description: Delete artifacts interactively with the App UI or programmatically with
  the W&B SDK/
displayed_sidebar: default
---

# 아티팩트 삭제하기

<head>
  <title>W&B 아티팩트 삭제하기</title>
</head>

앱 UI를 사용하여 대화식으로 아티팩트를 삭제하거나 W&B SDK를 사용하여 프로그래밍 방식으로 삭제할 수 있습니다. 아티팩트를 삭제하면 W&B는 해당 아티팩트를 *소프트 삭제* 표시합니다. 즉, 아티팩트가 삭제로 표시되지만 파일은 저장소에서 즉시 삭제되지 않습니다.

아티팩트의 내용은 소프트 삭제, 또는 삭제 대기 상태로 남아 있으며, 정기적으로 실행되는 가비지 컬렉션 프로세스가 삭제로 표시된 모든 아티팩트를 검토합니다. 아티팩트와 관련 파일이 이전 또는 이후의 아티팩트 버전과 사용되지 않는 경우 가비지 컬렉션 프로세스는 저장소에서 관련 파일을 삭제합니다.

이 페이지의 섹션에서는 특정 아티팩트 버전을 삭제하는 방법, 아티팩트 컬렉션을 삭제하는 방법, 에일리어스가 있는 아티팩트와 없는 아티팩트를 삭제하는 방법 등을 설명합니다. W&B의 TTL 정책을 사용하여 아티팩트가 삭제되는 시기를 예약할 수 있습니다. 자세한 정보는 [Artifact TTL 정책으로 데이터 유지 관리](./ttl.md)를 참조하세요.

:::note
TTL 정책으로 삭제가 예약된 아티팩트, W&B SDK로 삭제된 아티팩트 또는 W&B 앱 UI로 삭제된 아티팩트는 먼저 소프트 삭제됩니다. 소프트 삭제된 아티팩트는 하드 삭제되기 전에 가비지 컬렉션을 거칩니다.
:::

### 아티팩트 버전 삭제하기

아티팩트 버전을 삭제하려면:

1. 아티팩트의 이름을 선택합니다. 이렇게 하면 아티팩트 뷰가 확장되고 해당 아티팩트와 관련된 모든 아티팩트 버전이 나열됩니다.
2. 아티팩트 목록에서 삭제하려는 아티팩트 버전을 선택합니다.
3. 워크스페이스의 오른쪽에 있는 케밥 드롭다운을 선택합니다.
4. 삭제를 선택합니다.

아티팩트 버전은 [delete()](https://docs.wandb.ai/ref/python/artifact#delete) 메소드를 통해서도 프로그래밍 방식으로 삭제할 수 있습니다. 아래 예제를 참조하세요.

### 에일리어스가 있는 여러 아티팩트 버전 삭제하기

다음 코드 예제는 에일리어스가 있는 아티팩트를 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 엔티티, 프로젝트 이름 및 run ID를 제공하세요.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

아티팩트에 하나 이상의 에일리어스가 있는 경우 `delete_aliases` 파라미터를 불리언 값 `True`로 설정하여 에일리어스를 삭제합니다.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # 에일리어스가 하나 이상 있는 아티팩트를 삭제하려면
    # delete_aliases=True로 설정하세요.
    artifact.delete(delete_aliases=True)
```

### 특정 에일리어스가 있는 여러 아티팩트 버전 삭제하기

다음 코드는 특정 에일리어스가 있는 여러 아티팩트 버전을 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 엔티티, 프로젝트 이름 및 run ID를 제공하세요. 삭제 논리를 자신의 것으로 대체하세요:

```python
import wandb

runs = api.run("entity/project_name/run_id")

# 에일리어스 'v3'와 'v4'가 있는 아티팩트를 삭제합니다.
for artifact_version in runs.logged_artifacts():
    # 자신의 삭제 논리로 대체하세요.
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### 에일리어스가 없는 아티팩트의 모든 버전 삭제하기

다음 코드 조각은 에일리어스가 없는 아티팩트의 모든 버전을 삭제하는 방법을 보여줍니다. `wandb.Api`의 `project` 및 `entity` 키에 프로젝트 및 엔티티 이름을 제공하세요. `<>`를 아티팩트의 이름으로 대체하세요:

```python
import wandb

# wandb.Api 메소드를 사용할 때 엔티티와 프로젝트 이름을 제공하세요.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # 타입과 이름을 제공하세요.
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest'와 같은 에일리어스가 없는 버전을 정리합니다.
    # 참고: 여기에 원하는 삭제 논리를 넣을 수 있습니다.
    if len(v.aliases) == 0:
        v.delete()
```

### 아티팩트 컬렉션 삭제하기

아티팩트 컬렉션을 삭제하려면:

1. 삭제하려는 아티팩트 컬렉션으로 이동하여 마우스를 올리세요.
3. 아티팩트 컬렉션 이름 옆의 케밥 드롭다운을 선택하세요.
4. 삭제를 선택하세요.

또한 [delete()](../../ref/python/artifact.md#delete) 메소드를 사용하여 프로그래밍 방식으로 아티팩트 컬렉션을 삭제할 수 있습니다. `wandb.Api`에서 `project` 및 `entity` 키에 프로젝트 및 엔티티 이름을 제공하세요:

```python
import wandb

# wandb.Api 메소드를 사용할 때 엔티티와 프로젝트 이름을 제공하세요.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection("<artifact_type>", "entity/project/artifact_collection_name")
collection.delete()
```

## W&B가 호스팅되는 방식에 따른 가비지 컬렉션 활성화 방법
W&B의 공유 클라우드를 사용하는 경우 가비지 컬렉션이 기본적으로 활성화됩니다. W&B를 호스팅하는 방식에 따라 가비지 컬렉션을 활성화하기 위해 추가 조치를 취해야 할 수 있습니다. 여기에는 다음이 포함됩니다:

* `GORILLA_ARTIFACT_GC_ENABLED` 환경 변수를 true로 설정하세요: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/object-versioning) 또는 [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)와 같은 다른 저장소 제공업체를 사용하는 경우 버킷 버전 관리를 활성화하세요. Azure를 사용하는 경우 [소프트 삭제를 활성화하세요](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview).
  :::note
  Azure의 소프트 삭제는 다른 저장소 제공업체의 버킷 버전 관리와 동일합니다.
  :::

다음 표는 배포 유형에 따라 가비지 컬렉션을 활성화하기 위해 충족해야 하는 요구 사항을 설명합니다.

`X`는 요구 사항을 충족해야 함을 나타냅니다:

|                                                | 환경 변수              | 버전 관리 활성화  | 
| -----------------------------------------------| ------------------------| ----------------- | 
| 공유 클라우드                                 |                         |                   | 
| [보안 저장소 커넥터](../hosting/secure-storage-connector.md)가 있는 공유 클라우드|                         | X                 | 
| 전용 클라우드                                  |                         |                   | 
| [보안 저장소 커넥터](../hosting/secure-storage-connector.md)가 있는 전용 클라우드|                         | X                 | 
| 고객 관리 클라우드                             | X                       | X                 | 
| 고객 관리 온프레미스                           | X                       | X                 |
 

:::note
보안 저장소 커넥터는 현재 Google Cloud Platform 및 Amazon Web Services에서만 사용할 수 있습니다.
:::