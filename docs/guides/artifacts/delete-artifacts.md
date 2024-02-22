---
description: Delete artifacts interactively with the App UI or programmatically with
  the W&B SDK/
displayed_sidebar: default
---

# 아티팩트 삭제하기

<head>
  <title>W&B 아티팩트 삭제하기</title>
</head>

App UI 또는 W&B SDK를 사용하여 아티팩트를 대화식으로 또는 프로그래밍적으로 삭제할 수 있습니다. 아티팩트를 삭제하면 W&B는 해당 아티팩트를 *소프트 삭제*로 표시합니다. 즉, 아티팩트가 삭제로 표시되지만 파일은 스토리지에서 즉시 삭제되지 않습니다.

아티팩트의 내용은 소프트 삭제, 즉 삭제 대기 상태로 남아 있으며, 정기적으로 실행되는 가비지 컬렉션 프로세스가 삭제로 표시된 모든 아티팩트를 검토할 때까지 유지됩니다. 가비지 컬렉션 프로세스는 아티팩트와 그와 관련된 파일이 이전 또는 이후 아티팩트 버전에서 사용되지 않는 경우 스토리지에서 관련 파일을 삭제합니다.

이 페이지의 섹션에서는 특정 아티팩트 버전 삭제 방법, 아티팩트 컬렉션 삭제 방법, 별칭이 있는 아티팩트 및 없는 아티팩트를 삭제하는 방법 등을 설명합니다. W&B에서 아티팩트가 삭제되는 시기를 TTL 정책으로 예약할 수 있습니다. 자세한 정보는 [아티팩트 TTL 정책으로 데이터 보관 관리하기](./ttl.md)를 참조하세요.

:::note
TTL 정책으로 삭제가 예약된 아티팩트, W&B SDK로 삭제된 아티팩트 또는 W&B App UI로 삭제된 아티팩트는 먼저 소프트 삭제됩니다. 소프트 삭제된 아티팩트는 하드 삭제되기 전에 가비지 컬렉션을 거칩니다.
:::

### 아티팩트 버전 삭제하기

아티팩트 버전을 삭제하려면:

1. 아티팩트의 이름을 선택합니다. 이렇게 하면 아티팩트 뷰가 확장되고 해당 아티팩트와 관련된 모든 아티팩트 버전이 나열됩니다.
2. 아티팩트 목록에서 삭제하려는 아티팩트 버전을 선택합니다.
3. 워크스페이스 오른쪽에 있는 케밥 드롭다운을 선택합니다.
4. 삭제를 선택합니다.

아티팩트 버전은 또한 [delete()](https://docs.wandb.ai/ref/python/artifact#delete) 메서드를 통해 프로그래밍 방식으로 삭제될 수 있습니다. 아래 예시를 참조하세요.

### 별칭이 있는 여러 아티팩트 버전 삭제하기

다음 코드 예제는 별칭이 있는 아티팩트를 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 엔티티, 프로젝트 이름, 실행 ID를 제공합니다.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

`delete_aliases` 파라미터를 불리언 값 `True`로 설정하여 아티팩트에 하나 이상의 별칭이 있는 경우 별칭을 삭제합니다.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # 별칭이 한 개 이상 있는 아티팩트를 삭제하려면
    # delete_aliases=True로 설정합니다.
    artifact.delete(delete_aliases=True)
```

### 특정 별칭이 있는 여러 아티팩트 버전 삭제하기

다음 코드는 특정 별칭이 있는 여러 아티팩트 버전을 삭제하는 방법을 보여줍니다. 아티팩트를 생성한 엔티티, 프로젝트 이름, 실행 ID를 제공합니다. 삭제 로직을 자신의 것으로 대체하세요:

```python
import wandb

runs = api.run("entity/project_name/run_id")

# 별칭이 'v3' 및 'v4'인 아티팩트 삭제
for artifact_version in runs.logged_artifacts():
    # 자신의 삭제 로직으로 대체하세요.
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### 별칭이 없는 아티팩트의 모든 버전 삭제하기

다음 코드 조각은 별칭이 없는 아티팩트의 모든 버전을 삭제하는 방법을 보여줍니다. `wandb.Api`의 `project`와 `entity` 키에 대해 프로젝트와 엔티티의 이름을 제공합니다. `<>`를 아티팩트의 이름으로 교체하세요:

```python
import wandb

# wandb.Api 메서드를 사용할 때 자신의 엔티티와 프로젝트 이름을 제공하세요.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # 타입과 이름 제공
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest'와 같은 별칭이 없는 버전을 정리합니다.
    # 참고: 여기에 원하는 삭제 로직을 넣을 수 있습니다.
    if len(v.aliases) == 0:
        v.delete()
```

### 아티팩트 컬렉션 삭제하기

아티팩트 컬렉션을 삭제하려면:

1. 삭제하려는 아티팩트 컬렉션으로 이동하여 마우스를 가져갑니다.
3. 아티팩트 컬렉션 이름 옆에 있는 케밥 드롭다운을 선택합니다.
4. 삭제를 선택합니다.

[delete()](../../ref/python/artifact.md#delete) 메서드를 사용하여 프로그래밍 방식으로 아티팩트 컬렉션을 삭제할 수도 있습니다. `wandb.Api`의 `project`와 `entity` 키에 대해 프로젝트와 엔티티의 이름을 각각 제공하세요:

```python
import wandb

# wandb.Api 메서드를 사용할 때 자신의 엔티티와 프로젝트 이름을 제공하세요.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection("<artifact_type>", "entity/project/artifact_collection_name")
collection.delete()
```

## W&B 호스팅 방식에 따른 가비지 컬렉션 활성화 방법
W&B의 공유 클라우드를 사용하는 경우 가비지 컬렉션은 기본적으로 활성화됩니다. W&B를 호스팅하는 방식에 따라 가비지 컬렉션을 활성화하기 위해 추가 단계가 필요할 수 있으며, 여기에는 다음이 포함됩니다:

* `GORILLA_ARTIFACT_GC_ENABLED` 환경 변수를 true로 설정: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/object-versioning) 또는 [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)와 같은 다른 스토리지 제공업체를 사용하는 경우 버킷 버전 관리를 활성화합니다. Azure를 사용하는 경우 [소프트 삭제를 활성화](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)하세요.
  :::note
  Azure에서의 소프트 삭제는 다른 스토리지 제공업체에서의 버킷 버전 관리와 동등합니다.
  :::

다음 표는 배포 유형에 따라 가비지 컬렉션을 활성화하기 위해 충족해야 하는 요구 사항을 설명합니다.

`X`는 요구 사항을 충족해야 함을 나타냅니다:

|                                                | 환경 변수                | 버전 관리 활성화 | 
| -----------------------------------------------| ------------------------| ----------------- | 
| 공유 클라우드                                   |                         |                   | 
| [보안 스토리지 커넥터](../hosting/secure-storage-connector.md)가 있는 공유 클라우드|                         | X                 | 
| 데디케이티드 클라우드                                |                         |                   | 
| [보안 스토리지 커넥터](../hosting/secure-storage-connector.md)가 있는 데디케이티드 클라우드|                         | X                 | 
| 고객 관리 클라우드                         | X                       | X                 | 
| 고객 관리 온-프레미스                       | X                       | X                 |

:::note
보안 스토리지 커넥터는 현재 Google Cloud Platform과 Amazon Web Services에서만 사용할 수 있습니다.
:::