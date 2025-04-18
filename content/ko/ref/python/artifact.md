---
title: Artifact
menu:
  reference:
    identifier: ko-ref-python-artifact
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L96-L2410 >}}

데이터셋 및 모델 버전 관리를 위한 유연하고 가벼운 빌딩 블록입니다.

```python
Artifact(
    name: str,
    type: str,
    description: (str | None) = None,
    metadata: (dict[str, Any] | None) = None,
    incremental: bool = (False),
    use_as: (str | None) = None
) -> None
```

빈 W&B Artifact를 생성합니다. `add` 로 시작하는 메서드로 Artifact 내용을 채웁니다. Artifact에 원하는 파일이 모두 있으면 `wandb.log_artifact()` 를 호출하여 기록할 수 있습니다.

| Args |  |
| :--- | :--- |
|  `name` | Artifact에 대한 사람이 읽을 수 있는 이름입니다. W&B App UI 또는 프로그래밍 방식으로 특정 Artifact를 식별하는 데 이름을 사용합니다. `use_artifact` 퍼블릭 API를 사용하여 Artifact를 대화형으로 참조할 수 있습니다. 이름에는 문자, 숫자, 밑줄, 하이픈 및 점을 포함할 수 있습니다. 이름은 프로젝트에서 고유해야 합니다. |
|  `type` | Artifact의 유형입니다. Artifact의 유형을 사용하여 Artifact를 구성하고 구별합니다. 문자, 숫자, 밑줄, 하이픈 및 점을 포함하는 문자열을 사용할 수 있습니다. 일반적인 유형에는 `dataset` 또는 `model` 이 있습니다. Artifact를 W&B 모델 레지스트리에 연결하려면 유형 문자열 내에 `model` 을 포함합니다. |
|  `description` | Artifact에 대한 설명입니다. 모델 또는 데이터셋 Artifact의 경우 표준화된 팀 모델 또는 데이터셋 카드에 대한 문서를 추가합니다. `Artifact.description` 속성을 사용하여 프로그래밍 방식으로 또는 W&B App UI를 사용하여 프로그래밍 방식으로 Artifact의 설명을 봅니다. W&B는 W&B App에서 설명을 마크다운으로 렌더링합니다. |
|  `metadata` | Artifact에 대한 추가 정보입니다. 메타데이터를 키-값 쌍의 사전으로 지정합니다. 총 100개 이하의 키를 지정할 수 있습니다. |
|  `incremental` | 기존 Artifact를 수정하려면 대신 `Artifact.new_draft()` 메서드를 사용합니다. |
|  `use_as` | W&B Launch 특정 파라미터입니다. 일반적인 용도로는 권장하지 않습니다. |

| Returns |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

| Attributes |  |
| :--- | :--- |
|  `aliases` | Artifact 버전에 할당된 하나 이상의 의미적으로 친숙한 참조 또는 식별 "별칭" 목록입니다. 에일리어스는 프로그래밍 방식으로 참조할 수 있는 변경 가능한 참조입니다. W&B App UI 또는 프로그래밍 방식으로 Artifact의 에일리어스를 변경합니다. 자세한 내용은 [새 Artifact 버전 생성](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version)을 참조하십시오. |
|  `collection` | 이 Artifact가 검색된 컬렉션입니다. 컬렉션은 정렬된 Artifact 버전 그룹입니다. 이 Artifact가 포트폴리오 / 링크된 컬렉션에서 검색된 경우 Artifact 버전이 시작된 컬렉션이 아닌 해당 컬렉션이 반환됩니다. Artifact가 시작된 컬렉션을 소스 시퀀스라고 합니다. |
|  `commit_hash` | 이 Artifact가 커밋될 때 반환된 해시입니다. |
|  `created_at` | Artifact가 생성된 타임스탬프입니다. |
|  `description` | Artifact에 대한 설명입니다. |
|  `digest` | Artifact의 논리적 다이제스트입니다. 다이제스트는 Artifact 내용의 체크섬입니다. Artifact가 현재 `latest` 버전과 동일한 다이제스트를 갖는 경우 `log_artifact` 는 no-op입니다. |
|  `entity` | 보조 (포트폴리오) Artifact 컬렉션의 엔터티 이름입니다. |
|  `file_count` | 파일 수 (참조 포함) 입니다. |
|  `id` | Artifact의 ID입니다. |
|  `manifest` | Artifact의 manifest입니다. manifest는 모든 내용을 나열하며 Artifact가 기록된 후에는 변경할 수 없습니다. |
|  `metadata` | 사용자 정의 Artifact 메타데이터입니다. Artifact와 관련된 구조화된 데이터입니다. |
|  `name` | 보조 (포트폴리오) 컬렉션에서 Artifact 이름과 버전입니다. `{collection}:{alias}` 형식의 문자열입니다. Artifact가 저장되기 전에는 버전을 아직 알 수 없으므로 이름만 포함합니다. |
|  `project` | 보조 (포트폴리오) Artifact 컬렉션의 프로젝트 이름입니다. |
|  `qualified_name` | 보조 (포트폴리오) 컬렉션의 엔터티/프로젝트/이름입니다. |
|  `size` | 바이트 단위의 Artifact 총 크기입니다. 이 Artifact에서 추적하는 모든 참조를 포함합니다. |
|  `source_collection` | Artifact의 기본 (시퀀스) 컬렉션입니다. |
|  `source_entity` | 기본 (시퀀스) Artifact 컬렉션의 엔터티 이름입니다. |
|  `source_name` | 기본 (시퀀스) 컬렉션에서 Artifact 이름과 버전입니다. `{collection}:{alias}` 형식의 문자열입니다. Artifact가 저장되기 전에는 버전을 아직 알 수 없으므로 이름만 포함합니다. |
|  `source_project` | 기본 (시퀀스) Artifact 컬렉션의 프로젝트 이름입니다. |
|  `source_qualified_name` | 기본 (시퀀스) 컬렉션의 엔터티/프로젝트/이름입니다. |
|  `source_version` | 기본 (시퀀스) 컬렉션에서 Artifact의 버전입니다. `v{number}` 형식의 문자열입니다. |
|  `state` | Artifact의 상태입니다. "PENDING", "COMMITTED" 또는 "DELETED" 중 하나입니다. |
|  `tags` | 이 Artifact 버전에 할당된 하나 이상의 태그 목록입니다. |
|  `ttl` | Artifact의 TTL (Time-To-Live) 정책입니다. Artifact는 TTL 정책의 기간이 경과한 직후에 삭제됩니다. `None` 으로 설정하면 Artifact는 TTL 정책을 비활성화하고 팀 기본 TTL이 있더라도 삭제가 예약되지 않습니다. 팀 관리자가 기본 TTL을 정의하고 Artifact에 사용자 지정 정책이 설정되어 있지 않으면 Artifact는 팀 기본값에서 TTL 정책을 상속합니다. |
|  `type` | Artifact의 유형입니다. 일반적인 유형에는 `dataset` 또는 `model` 이 있습니다. |
|  `updated_at` | Artifact가 마지막으로 업데이트된 시간입니다. |
|  `url` | Artifact의 URL을 생성합니다. |
|  `version` | 보조 (포트폴리오) 컬렉션에서 Artifact의 버전입니다. |

## Methods

### `add`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1487-L1578)

```python
add(
    obj: WBValue,
    name: StrPath,
    overwrite: bool = (False)
) -> ArtifactManifestEntry
```

wandb.WBValue `obj` 를 Artifact에 추가합니다.

| Args |  |
| :--- | :--- |
|  `obj` | 추가할 오브젝트입니다. 현재 Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D 중 하나를 지원합니다. |
|  `name` | 오브젝트를 추가할 Artifact 내의 경로입니다. |
|  `overwrite` | True인 경우 동일한 파일 경로를 가진 기존 오브젝트를 덮어씁니다 (해당하는 경우). |

| Returns |  |
| :--- | :--- |
|  추가된 manifest 항목 |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` | Artifact 버전이 완료되었으므로 현재 Artifact 버전을 변경할 수 없습니다. 대신 새 Artifact 버전을 기록합니다. |

### `add_dir`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1342-L1402)

```python
add_dir(
    local_path: str,
    name: (str | None) = None,
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable"
) -> None
```

로컬 디렉토리를 Artifact에 추가합니다.

| Args |  |
| :--- | :--- |
|  `local_path` | 로컬 디렉토리의 경로입니다. |
|  `name` | Artifact 내의 하위 디렉토리 이름입니다. 지정한 이름은 Artifact의 `type` 에 따라 중첩된 W&B App UI에 나타납니다. 기본값은 Artifact의 루트입니다. |
|  `skip_cache` | `True` 로 설정하면 W&B는 업로드하는 동안 파일을 캐시로 복사/이동하지 않습니다. |
|  `policy` | "mutable" | "immutable". 기본적으로 "mutable" "mutable": 업로드 중에 손상을 방지하기 위해 파일의 임시 복사본을 만듭니다. "immutable": 보호를 비활성화하고 사용자가 파일을 삭제하거나 변경하지 않도록 합니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` | Artifact 버전이 완료되었으므로 현재 Artifact 버전을 변경할 수 없습니다. 대신 새 Artifact 버전을 기록합니다. |
|  `ValueError` | 정책은 "mutable" 또는 "immutable"이어야 합니다. |

### `add_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1289-L1340)

```python
add_file(
    local_path: str,
    name: (str | None) = None,
    is_tmp: (bool | None) = (False),
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable",
    overwrite: bool = (False)
) -> ArtifactManifestEntry
```

로컬 파일을 Artifact에 추가합니다.

| Args |  |
| :--- | :--- |
|  `local_path` | 추가할 파일의 경로입니다. |
|  `name` | 추가할 파일에 사용할 Artifact 내의 경로입니다. 기본값은 파일의 기본 이름입니다. |
|  `is_tmp` | true이면 충돌을 피하기 위해 파일 이름이 결정적으로 변경됩니다. |
|  `skip_cache` | `True` 인 경우 W&B는 업로드 후 파일을 캐시로 복사하지 않습니다. |
|  `policy` | 기본적으로 "mutable"로 설정됩니다. "mutable"로 설정하면 업로드 중에 손상을 방지하기 위해 파일의 임시 복사본을 만듭니다. "immutable"로 설정하면 보호를 비활성화하고 사용자가 파일을 삭제하거나 변경하지 않도록 합니다. |
|  `overwrite` | `True` 인 경우 파일이 이미 있으면 덮어씁니다. |

| Returns |  |
| :--- | :--- |
|  추가된 manifest 항목입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` | Artifact 버전이 완료되었으므로 현재 Artifact 버전을 변경할 수 없습니다. 대신 새 Artifact 버전을 기록합니다. |
|  `ValueError` | 정책은 "mutable" 또는 "immutable"이어야 합니다. |

### `add_reference`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1404-L1485)

```python
add_reference(
    uri: (ArtifactManifestEntry | str),
    name: (StrPath | None) = None,
    checksum: bool = (True),
    max_objects: (int | None) = None
) -> Sequence[ArtifactManifestEntry]
```

URI로 표시된 참조를 Artifact에 추가합니다.

Artifact에 추가하는 파일 또는 디렉토리와 달리 참조는 W&B에 업로드되지 않습니다. 자세한 내용은 [외부 파일 추적](https://docs.wandb.ai/guides/artifacts/track-external-files)을 참조하십시오.

기본적으로 다음 스키마가 지원됩니다.

- http(s): 파일 크기 및 다이제스트는 서버에서 반환된 `Content-Length` 및 `ETag` 응답 헤더에 의해 추론됩니다.
- s3: 체크섬 및 크기는 오브젝트 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우 버전 ID도 추적됩니다.
- gs: 체크섬 및 크기는 오브젝트 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우 버전 ID도 추적됩니다.
- https, 도메인 일치 `*.blob.core.windows.net` (Azure): 체크섬 및 크기는 Blob 메타데이터에서 가져옵니다. 스토리지 계정 버전 관리가 활성화된 경우 버전 ID도 추적됩니다.
- file: 체크섬 및 크기는 파일 시스템에서 가져옵니다. 이 스키마는 추적하고 싶지만 반드시 업로드할 필요는 없는 파일이 포함된 NFS 공유 또는 기타 외부 탑재 볼륨이 있는 경우에 유용합니다.

다른 스키마의 경우 다이제스트는 URI의 해시일 뿐이고 크기는 비어 있습니다.

| Args |  |
| :--- | :--- |
|  `uri` | 추가할 참조의 URI 경로입니다. URI 경로는 다른 Artifact의 항목에 대한 참조를 저장하기 위해 `Artifact.get_entry` 에서 반환된 오브젝트일 수 있습니다. |
|  `name` | 이 참조 내용을 배치할 Artifact 내의 경로입니다. |
|  `checksum` | 참조 URI에 있는 리소스의 체크섬을 계산할지 여부입니다. 자동 무결성 유효성 검사를 활성화하므로 체크섬 계산을 강력히 권장합니다. 체크섬 계산을 비활성화하면 Artifact 생성이 빨라지지만 참조 디렉토리가 반복되지 않으므로 디렉토리의 오브젝트가 Artifact에 저장되지 않습니다. 참조 오브젝트를 추가할 때 `checksum=False` 를 설정하는 것이 좋습니다. 이 경우 참조 URI가 변경된 경우에만 새 버전이 생성됩니다. |
|  `max_objects` | 디렉토리 또는 버킷 스토어 접두사를 가리키는 참조를 추가할 때 고려할 최대 오브젝트 수입니다. 기본적으로 Amazon S3, GCS, Azure 및 로컬 파일에 허용되는 최대 오브젝트 수는 10,000,000개입니다. 다른 URI 스키마에는 최대값이 없습니다. |

| Returns |  |
| :--- | :--- |
|  추가된 manifest 항목입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` | Artifact 버전이 완료되었으므로 현재 Artifact 버전을 변경할 수 없습니다. 대신 새 Artifact 버전을 기록합니다. |

### `checkout`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1993-L2021)

```python
checkout(
    root: (str | None) = None
) -> str
```

지정된 루트 디렉토리를 Artifact 내용으로 바꿉니다.

경고: 이렇게 하면 Artifact에 포함되지 않은 `root` 의 모든 파일이 삭제됩니다.

| Args |  |
| :--- | :--- |
|  `root` | 이 Artifact의 파일로 대체할 디렉토리입니다. |

| Returns |  |
| :--- | :--- |
|  체크아웃된 내용의 경로입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2131-L2150)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

Artifact 및 해당 파일을 삭제합니다.

링크된 Artifact (즉, 포트폴리오 컬렉션의 멤버) 에서 호출된 경우 링크만 삭제되고 소스 Artifact는 영향을 받지 않습니다.

| Args |  |
| :--- | :--- |
|  `delete_aliases` | `True` 로 설정하면 Artifact와 연결된 모든 에일리어스를 삭제합니다. 그렇지 않으면 Artifact에 기존 에일리어스가 있는 경우 예외가 발생합니다. Artifact가 링크된 경우 (즉, 포트폴리오 컬렉션의 멤버) 이 파라미터는 무시됩니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |

### `download`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1756-L1807)

```python
download(
    root: (StrPath | None) = None,
    allow_missing_references: bool = (False),
    skip_cache: (bool | None) = None,
    path_prefix: (StrPath | None) = None
) -> FilePathStr
```

Artifact 내용을 지정된 루트 디렉토리에 다운로드합니다.

`root` 내에 있는 기존 파일은 수정되지 않습니다. `root` 의 내용을 Artifact와 정확히 일치시키려면 `download` 를 호출하기 전에 명시적으로 `root` 를 삭제하십시오.

| Args |  |
| :--- | :--- |
|  `root` | W&B가 Artifact 파일을 저장하는 디렉토리입니다. |
|  `allow_missing_references` | `True` 로 설정하면 참조된 파일을 다운로드하는 동안 잘못된 참조 경로는 무시됩니다. |
|  `skip_cache` | `True` 로 설정하면 다운로드할 때 Artifact 캐시가 건너뛰고 W&B는 각 파일을 기본 루트 또는 지정된 다운로드 디렉토리에 다운로드합니다. |
|  `path_prefix` | 지정된 경우 지정된 접두사로 시작하는 경로가 있는 파일만 다운로드됩니다. 유닉스 형식 (forward slash) 을 사용합니다. |

| Returns |  |
| :--- | :--- |
|  다운로드된 내용의 경로입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |
|  `RuntimeError` | Artifact를 오프라인 모드에서 다운로드하려고 시도한 경우. |

### `file`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2063-L2087)

```python
file(
    root: (str | None) = None
) -> StrPath
```

단일 파일 Artifact를 `root` 로 지정한 디렉토리에 다운로드합니다.

| Args |  |
| :--- | :--- |
|  `root` | 파일을 저장할 루트 디렉토리입니다. 기본값은 './artifacts/self.name/' 입니다. |

| Returns |  |
| :--- | :--- |
|  다운로드한 파일의 전체 경로입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |
|  `ValueError` | Artifact에 둘 이상의 파일이 포함된 경우. |

### `files`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2089-L2106)

```python
files(
    names: (list[str] | None) = None,
    per_page: int = 50
) -> ArtifactFiles
```

이 Artifact에 저장된 모든 파일을 반복합니다.

| Args |  |
| :--- | :--- |
|  `names` | 나열할 Artifact의 루트를 기준으로 하는 파일 이름 경로입니다. |
|  `per_page` | 요청당 반환할 파일 수입니다. |

| Returns |  |
| :--- | :--- |
|  `File` 오브젝트를 포함하는 반복기입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |

### `finalize`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L902-L910)

```python
finalize() -> None
```

Artifact 버전을 완료합니다.

Artifact가 특정 Artifact 버전으로 기록되므로 Artifact 버전이 완료되면 Artifact 버전을 수정할 수 없습니다. Artifact에 더 많은 데이터를 기록하려면 새 Artifact 버전을 만듭니다. `log_artifact` 로 Artifact를 기록하면 Artifact가 자동으로 완료됩니다.

### `get`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1673-L1718)

```python
get(
    name: str
) -> (WBValue | None)
```

Artifact 상대 `name` 에 있는 WBValue 오브젝트를 가져옵니다.

| Args |  |
| :--- | :--- |
|  `name` | 검색할 Artifact 상대 이름입니다. |

| Returns |  |
| :--- | :--- |
|  `wandb.log()` 로 기록하고 W&B UI에서 시각화할 수 있는 W&B 오브젝트. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않았거나 run이 오프라인인 경우 |

### `get_added_local_path_name`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1720-L1732)

```python
get_added_local_path_name(
    local_path: str
) -> (str | None)
```

로컬 파일 시스템 경로로 추가된 파일의 Artifact 상대 이름을 가져옵니다.

| Args |  |
| :--- | :--- |
|  `local_path` | Artifact 상대 이름으로 해석할 로컬 경로입니다. |

| Returns |  |
| :--- | :--- |
|  Artifact 상대 이름. |

### `get_entry`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1652-L1671)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

지정된 이름으로 항목을 가져옵니다.

| Args |  |
| :--- | :--- |
|  `name` | 가져올 Artifact 상대 이름입니다. |

| Returns |  |
| :--- | :--- |
|  `W&B` 오브젝트. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않았거나 run이 오프라인인 경우. |
|  `KeyError` | Artifact에 지정된 이름의 항목이 포함되어 있지 않은 경우. |

### `get_path`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1644-L1650)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

더 이상 사용되지 않습니다. `get_entry(name)` 을 사용하십시오.

### `is_draft`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L912-L917)

```python
is_draft() -> bool
```

Artifact가 저장되지 않았는지 확인합니다.

반환 값: Boolean. Artifact가 저장된 경우 `False` 입니다. Artifact가 저장되지 않은 경우 `True` 입니다.

### `json_encode`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2338-L2345)

```python
json_encode() -> dict[str, Any]
```

Artifact를 JSON 형식으로 인코딩하여 반환합니다.

| Returns |  |
| :--- | :--- |
| Artifact의 속성을 나타내는 `string` 키가 있는 `dict` 입니다. |

### `link`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2177-L2205)

```python
link(
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
```

이 Artifact를 포트폴리오 (승격된 Artifact 컬렉션) 에 연결합니다.

| Args |  |
| :--- | :--- |
|  `target_path` | 프로젝트 내의 포트폴리오 경로입니다. 대상 경로는 `{portfolio}`, `{project}/{portfolio}` 또는 `{entity}/{project}/{portfolio}` 중 하나의 스키마를 준수해야 합니다. Artifact를 일반적인 프로젝트 내의 포트폴리오가 아닌 모델 레지스트리에 연결하려면 `target_path` 를 `{"model-registry"}/{Registered Model Name}` 또는 `{entity}/{"model-registry"}/{Registered Model Name}` 스키마로 설정합니다. |
|  `aliases` | 지정된 포트폴리오 내에서 Artifact를 고유하게 식별하는 문자열 목록입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |

### `logged_by`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2294-L2336)

```python
logged_by() -> (Run | None)
```

Artifact를 원래 기록한 W&B run을 가져옵니다.

| Returns |  |
| :--- | :--- |
| Artifact를 원래 기록한 W&B run의 이름입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |

### `new_draft`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L424-L457)

```python
new_draft() -> Artifact
```

커밋된 이 Artifact와 동일한 내용으로 새 초안 Artifact를 만듭니다.

기존 Artifact를 수정하면 "증분 Artifact"라고 하는 새 Artifact 버전이 생성됩니다. 반환된 Artifact는 새 버전으로 확장, 수정 및 기록할 수 있습니다.

| Returns |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |

### `new_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1246-L1287)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "x",
    encoding: (str | None) = None
) -> Iterator[IO]
```

새 임시 파일을 열고 Artifact에 추가합니다.

| Args |  |
| :--- | :--- |
|  `name` | Artifact에 추가할 새 파일의 이름입니다. |
|  `mode` | 새 파일을 여는 데 사용할 파일 액세스 모드입니다. |
|  `encoding` | 새 파일을 여는 데 사용되는 인코딩입니다. |

| Returns |  |
| :--- | :--- |
|  쓸 수 있는 새 파일 오브젝트입니다. 닫으면 파일이 자동으로 Artifact에 추가됩니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` | Artifact 버전이 완료되었으므로 현재 Artifact 버전을 변경할 수 없습니다. 대신 새 Artifact 버전을 기록합니다. |

### `remove`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1614-L1642)

```python
remove(
    item: (StrPath | ArtifactManifestEntry)
) -> None
```

Artifact에서 항목을 제거합니다.

| Args |  |
| :--- | :--- |
|  `item` | 제거할 항목입니다. 특정 manifest 항목 또는 Artifact 상대 경로의 이름일 수 있습니다. 항목이 디렉토리를 일치시키면 해당 디렉토리의 모든 항목이 제거됩니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` | Artifact 버전이 완료되었으므로 현재 Artifact 버전을 변경할 수 없습니다. 대신 새 Artifact 버전을 기록합니다. |
|  `FileNotFoundError` | Artifact에서 항목을 찾을 수 없는 경우. |

### `save`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L922-L961)

```python
save(
    project: (str | None) = None,
    settings: (wandb.Settings | None) = None
) -> None
```

Artifact에 대한 모든 변경 사항을 유지합니다.

현재 run에 있는 경우 해당 run은 이 Artifact를 기록합니다. 현재 run에 없는 경우 이 Artifact를 추적하기 위해 "auto" 유형의 run이 생성됩니다.

| Args |  |
| :--- | :--- |
|  `project` | run이 이미 컨텍스트에 없는 경우 Artifact에 사용할 프로젝트입니다. |
|  `settings` | 자동 run을 초기화할 때 사용할 설정 오브젝트입니다. 가장 일반적으로 테스트 하니스에서 사용됩니다. |

### `unlink`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2207-L2222)

```python
unlink() -> None
```

이 Artifact가 현재 포트폴리오 (승격된 Artifact 컬렉션) 의 멤버인 경우 연결을 해제합니다.

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |
|  `ValueError` | Artifact가 링크되지 않은 경우, 즉 포트폴리오 컬렉션의 멤버가 아닌 경우. |

### `used_by`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2248-L2292)

```python
used_by() -> list[Run]
```

이 Artifact를 사용한 run 목록을 가져옵니다.

| Returns |  |
| :--- | :--- |
|  `Run` 오브젝트 목록입니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |

### `verify`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2023-L2061)

```python
verify(
    root: (str | None) = None
) -> None
```

Artifact 내용이 manifest와 일치하는지 확인합니다.

디렉토리의 모든 파일은 체크섬이 계산되고 체크섬은 Artifact의 manifest와 상호 참조됩니다. 참조는 확인되지 않습니다.

| Args |  |
| :--- | :--- |
|  `root` | 확인할 디렉토리입니다. None Artifact인 경우 './artifacts/self.name/' 에 다운로드됩니다. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않은 경우. |
|  `ValueError` | 확인에 실패한 경우. |

### `wait`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L971-L995)

```python
wait(
    timeout: (int | None) = None
) -> Artifact
```

필요한 경우 이 Artifact가 기록을 완료할 때까지 기다립니다.

| Args |  |
| :--- | :--- |
|  `timeout` | 대기 시간 (초) 입니다. |

| Returns |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

### `__getitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1216-L1228)

```python
__getitem__(
    name: str
) -> (WBValue | None)
```

Artifact 상대 `name` 에 있는 WBValue 오브젝트를 가져옵니다.

| Args |  |
| :--- | :--- |
|  `name` | 가져올 Artifact 상대 이름입니다. |

| Returns |  |
| :--- | :--- |
|  `wandb.log()` 로 기록하고 W&B UI에서 시각화할 수 있는 W&B 오브젝트. |

| Raises |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` | Artifact가 기록되지 않았거나 run이 오프라인인 경우. |

### `__setitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1230-L1244)

```python
__setitem__(
    name: str,
    item: WBValue
) -> ArtifactManifestEntry
```

경로 `name` 에서 Artifact에 `item` 을 추가합니다.

| Args |  |
| :--- | :--- |
|  `name` | 오브젝트를 추가할 Artifact 내의 경로입니다. |
|  `item` | 추가할 오브젝트입니다. |

| Returns |  |
| :--- | :--- |
|  추가된 manifest 항목 |

| Raises |  |
| :--- | :--- |
|  `ArtifactFinalizedError` | Artifact 버전이 완료되었으므로 현재 Artifact 버전을 변경할 수 없습니다. 대신 새 Artifact 버전을 기록합니다. |