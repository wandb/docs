# Artifact

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L95-L2407' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

데이터셋과 모델 버전 관리에 유연하고 가벼운 빌딩 블록입니다.

```python
Artifact(
    name: str,
    type: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    incremental: bool = (False),
    use_as: Optional[str] = None
) -> None
```

빈 W&B 아티팩트를 생성합니다. `add`로 시작하는 메소드를 사용하여 아티팩트의 내용을 채웁니다. 아티팩트에 모든 필요한 파일이 추가되면, `wandb.log_artifact()`를 호출하여 로그할 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  아티팩트를 식별하기 위한 사람이 읽을 수 있는 이름입니다. 이 이름을 사용하여 W&B 앱 UI나 프로그램적으로 특정 아티팩트를 식별할 수 있습니다. 인터렉티브하게 `use_artifact` Public API를 통해 아티팩트를 참조할 수 있습니다. 이름에는 문자, 숫자, 밑줄, 하이픈, 점을 포함할 수 있으며, 프로젝트 내에서 고유해야 합니다. |
|  `type` |  아티팩트의 유형입니다. 아티팩트의 유형을 사용하여 아티팩트를 정리하고 구분할 수 있습니다. 문자, 숫자, 밑줄, 하이픈, 점을 포함하는 임의의 문자열을 사용할 수 있습니다. 일반적인 유형은 `dataset` 또는 `model`입니다. W&B 모델 레지스트리에 아티팩트를 연결하려면 유형 문자열에 `model`을 포함하세요. |
|  `description` |  아티팩트에 대한 설명입니다. 모델이나 데이터셋 아티팩트의 경우 팀 모델이나 데이터셋 카드에 대한 문서를 추가하세요. 아티팩트의 설명을 `Artifact.description` 속성을 통해 프로그램적으로 보거나 W&B 앱 UI에서 확인할 수 있습니다. W&B 앱에서는 설명을 마크다운으로 렌더링합니다. |
|  `metadata` |  아티팩트에 대한 추가 정보입니다. 메타데이터는 키-값 쌍의 사전으로 지정하십시오. 키의 총 개수는 100개를 넘을 수 없습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

| 속성 |  |
| :--- | :--- |
|  `aliases` |  아티팩트 버전에 할당된 하나 이상의 의미론적으로 친숙한 참조 또는 식별 "닉네임" 목록입니다. 에일리어스는 프로그램적으로 참조할 수 있는 변경 가능한 참조입니다. W&B 앱 UI나 프로그램적으로 아티팩트의 에일리어스를 변경하세요. 자세한 내용은 [새로운 아티팩트 버전 생성](/guides/artifacts/create-a-new-artifact-version)을 참조하세요. |
|  `collection` |  아티팩트를 가져온 컬렉션입니다. 컬렉션은 아티팩트 버전의 정렬된 그룹입니다. 이 아티팩트가 포트폴리오/링크된 컬렉션에서 검색된 경우, 아티팩트 버전이 유래한 컬렉션이 아닌 해당 컬렉션이 반환됩니다. 아티팩트가 유래한 컬렉션은 '소스 시퀀스'라고 알려져 있습니다. |
|  `commit_hash` |  이 아티팩트가 커밋될 때 반환된 해시입니다. |
|  `created_at` |  아티팩트가 생성된 타임스탬프입니다. |
|  `description` |  아티팩트에 대한 설명입니다. |
|  `digest` |  아티팩트의 논리적 다이제스트입니다. 다이제스트는 아티팩트 내용의 체크섬입니다. 만약 현재 `latest` 버전과 다이제스트가 동일한 아티팩트가 있다면 `log_artifact`는 아무 작업을 하지 않습니다. |
|  `entity` |  이차적인(포트폴리오) 아티팩트 컬렉션의 엔티티 이름입니다. |
|  `file_count` |  파일(참조 포함)의 수입니다. |
|  `id` |  아티팩트의 ID입니다. |
|  `manifest` |  아티팩트의 매니페스트입니다. 매니페스트는 모든 내용을 나열하며, 아티팩트가 로그되면 변경할 수 없습니다. |
|  `metadata` |  사용자 정의 아티팩트 메타데이터입니다. 아티팩트와 연관된 구조화된 데이터입니다. |
|  `name` |  이차(포트폴리오) 컬렉션에서의 아티팩트 이름과 버전입니다. 이는 `{collection}:{alias}` 형식을 가진 문자열입니다. 아티팩트가 저장되기 전에 버전은 아직 알려지지 않았기 때문에 이름만 포함하고 있습니다. |
|  `project` |  이차(포트폴리오) 아티팩트 컬렉션의 프로젝트 이름입니다. |
|  `qualified_name` |  이차(포트폴리오) 컬렉션의 엔티티/프로젝트/이름입니다. |
|  `size` |  아티팩트의 총 크기(바이트)입니다. 이 아티팩트가 추적하는 참조를 포함합니다. |
|  `source_collection` |  아티팩트의 주요(시퀀스) 컬렉션입니다. |
|  `source_entity` |  주요(시퀀스) 아티팩트 컬렉션의 엔티티 이름입니다. |
|  `source_name` |  주요(시퀀스) 컬렉션에서의 아티팩트 이름과 버전입니다. 이는 `{collection}:{alias}` 형식을 가진 문자열입니다. 아티팩트가 저장되기 전에 버전은 아직 알려지지 않았기 때문에 이름만 포함하고 있습니다. |
|  `source_project` |  주요(시퀀스) 아티팩트 컬렉션의 프로젝트 이름입니다. |
|  `source_qualified_name` |  주요(시퀀스) 컬렉션의 엔티티/프로젝트/이름입니다. |
|  `source_version` |  주요(시퀀스) 컬렉션에서의 아티팩트 버전입니다. 이는 `v{number}` 형식을 가진 문자열입니다. |
|  `state` |  아티팩트의 상태입니다. "PENDING", "COMMITTED", "DELETED" 중 하나입니다. |
|  `tags` |  이 아티팩트 버전에 할당된 하나 이상의 태그 목록입니다. |
|  `ttl` |  아티팩트의 생존 시간(TTL) 정책입니다. TTL 정책의 기간이 지나면 아티팩트는 곧 삭제됩니다. `None`으로 설정하면 TTL 정책이 비활성화되며 팀 기본 TTL이 있더라도 아티팩트는 삭제 대상에 포함되지 않습니다. 아티팩트는 TTL 관리자가 기본 TTL을 설정하고 아티팩트에 사용자 정의 정책이 설정되지 않은 경우 기본 TTL 정책을 상속받습니다. |
|  `type` |  아티팩트의 유형입니다. 일반적인 유형은 `dataset` 또는 `model`입니다. |
|  `updated_at` |  아티팩트가 마지막으로 업데이트된 시간입니다. |
|  `version` |  이차(포트폴리오) 컬렉션에서의 아티팩트 버전입니다. |

## 메소드

### `add`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1397-L1494)

```python
add(
    obj: data_types.WBValue,
    name: StrPath
) -> ArtifactManifestEntry
```

wandb.WBValue `obj`를 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `obj` |  추가할 오브젝트입니다. 현재 Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D 중 하나를 지원합니다. |
|  `name` |  오브젝트를 추가할 아티팩트 내 경로입니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목 |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 이미 확정되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |

### `add_dir`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1252-L1312)

```python
add_dir(
    local_path: str,
    name: Optional[str] = None,
    skip_cache: Optional[bool] = (False),
    policy: Optional[Literal['mutable', 'immutable']] = "mutable"
) -> None
```

로컬 디렉토리를 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `local_path` |  로컬 디렉토리 경로입니다. |
|  `name` |  아티팩트 내의 하위 디렉토리 이름입니다. W&B 앱 UI에서 아티팩트의 `type`별로 중첩되어 표시됩니다. 기본값은 아티팩트의 루트입니다. |
|  `skip_cache` |  `True`로 설정하면, W&B는 업로드 중에 파일을 캐시에 복사/이동하지 않습니다. |
|  `policy` |  "mutable" | "immutable". 기본적으로 "mutable" "mutable": 업로드 중 손상을 방지하기 위해 파일의 임시 복사본을 만듭니다. "immutable": 보호를 비활성화하고 사용자가 파일을 삭제하거나 변경하지 않도록 합니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 이미 확정되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |
|  `ValueError` |  정책은 "mutable" 또는 "immutable"이어야 합니다. |

### `add_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1206-L1250)

```python
add_file(
    local_path: str,
    name: Optional[str] = None,
    is_tmp: Optional[bool] = (False),
    skip_cache: Optional[bool] = (False),
    policy: Optional[Literal['mutable', 'immutable']] = "mutable"
) -> ArtifactManifestEntry
```

로컬 파일을 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `local_path` |  추가할 파일의 경로입니다. |
|  `name` |  추가할 파일의 아티팩트 내 경로입니다. 기본값은 파일의 basename입니다. |
|  `is_tmp` |  true인 경우, 충돌을 피하기 위해 파일이 결정론적으로 이름 변경됩니다. |
|  `skip_cache` |  `True`로 설정하면, W&B는 업로드 후 파일을 캐시에 복사하지 않습니다. |
|  `policy` |  "mutable" | "immutable". 기본적으로 "mutable" "mutable": 업로드 중 손상을 방지하기 위해 파일의 임시 복사본을 만듭니다. "immutable": 보호를 비활성화하고 사용자가 파일을 삭제하거나 변경하지 않도록 합니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목 |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 이미 확정되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |
|  `ValueError` |  정책은 "mutable" 또는 "immutable"이어야 합니다. |

### `add_reference`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1314-L1395)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

URI로 표시된 참조를 아티팩트에 추가합니다.

아티팩트에 파일이나 디렉토리를 추가하는 것과 달리, 참조는 W&B에 업로드되지 않습니다. 자세한 정보는 [외부 파일 추적](/guides/artifacts/track-external-files)을 참조하세요.

기본적으로 다음 스키마가 지원됩니다:

- http(s): 파일의 크기와 다이제스트는 서버에서 반환받은 `Content-Length`와 `ETag` 응답 헤더에 의해 추론됩니다.
- s3: 체크섬과 크기는 오브젝트 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우, 버전 ID도 추적됩니다.
- gs: 체크섬과 크기는 오브젝트 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우, 버전 ID도 추적됩니다.
- https, 'blob.core.windows.net' 도메인 일치(Azure): 체크섬과 크기는 blob 메타데이터에서 가져옵니다. 스토리지 계정 버전 관리가 활성화된 경우, 버전 ID도 추적됩니다.
- file: 체크섬과 크기는 파일 시스템에서 가져옵니다. 외부에 마운트된 볼륨에 포함된 파일을 추적하지만 반드시 업로드할 필요는 없는 경우에 유용합니다.

다른 스키마의 경우, 다이제스트는 URI의 해시일 뿐이고 크기는 비워둡니다.

| 인수 |  |
| :--- | :--- |
|  `uri` |  추가할 참조의 URI 경로입니다. URI 경로는 다른 아티팩트의 엔트리에 대한 참조를 저장하기 위해 `Artifact.get_entry`에서 반환된 오브젝트일 수 있습니다. |
|  `name` |  이 참조의 내용을 넣을 아티팩트 내 경로입니다. |
|  `checksum` |  참조 URI에 위치한 리소스의 체크섬을 수행할지 여부입니다. 체크섬 처리는 자동 무결성 검증을 가능하게 하므로 강력히 권장됩니다. 체크섬을 비활성화하면 아티팩트 생성 속도가 빨라지지만 참조 디렉토리를 탐색하지 않아 디렉토리 내 오브젝트가 아티팩트에 저장되지 않습니다. 참조 오브젝트를 추가할 때 `checksum=False`로 설정하는 것을 권장하며, 이 경우 참조 URI가 변경될 때만 새 버전이 생성됩니다. |
|  `max_objects` |  디렉토리 또는 버킷 저장소 접두사에 대한 참조를 추가할 때 고려할 최대 오브젝트 수입니다. 기본적으로 Amazon S3, GCS, Azure, 로컬 파일에 대해 허용되는 최대 오브젝트 수는 10,000,000입니다. 다른 URI 스키마에는 최대값이 없습니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목들. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 이미 확정되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |

### `checkout`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1922-L1951)

```python
checkout(
    root: Optional[str] = None
) -> str
```

지정된 루트 디렉토리의 내용을 아티팩트의 내용으로 교체합니다.

경고: 아티팩트에 포함되지 않은 `root`의 모든 파일은 삭제됩니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  이 아티팩트의 파일로 교체할 디렉토리입니다. |

| 반환값 |  |
| :--- | :--- |
|  체크아웃된 콘텐츠의 경로입니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L2063-L2082)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

아티팩트와 해당 파일을 삭제합니다.

링크된 아티팩트(즉, 포트폴리오 컬렉션의 멤버)에 대해 호출된 경우: 링크만 삭제되고 소스 아티팩트는 영향을 받지 않습니다.

| 인수 |  |
| :--- | :--- |
|  `delete_aliases` |  `True`로 설정하면 아티팩트와 연관된 모든 에일리어스를 삭제합니다. 그렇지 않으면, 아티팩트에 기존 에일리어스가 있는 경우 예외가 발생합니다. 아티팩트가 링크된 경우(즉, 포트폴리오 컬렉션의 멤버)에는 이 인수가 무시됩니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1674-L1726)

```python
download(
    root: Optional[StrPath] = None,
    allow_missing_references: bool = (False),
    skip_cache: Optional[bool] = None,
    path_prefix: Optional[StrPath] = None
) -> FilePathStr
```

지정된 루트 디렉토리에 아티팩트의 내용을 다운로드합니다.

`root` 내 기존 파일은 수정되지 않습니다. `download`를 호출하기 전에 `root`를 명시적으로 삭제해야 `root`의 내용이 아티팩트와 정확히 일치합니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  W&B가 아티팩트의 파일을 저장하는 디렉토리입니다. |
|  `allow_missing_references` |  `True`로 설정하면, 참조 파일을 다운로드하는 동안 잘못된 참조 경로가 무시됩니다. |
|  `skip_cache` |  `True`로 설정하면, 아티팩트를 다운로드할 때 캐시를 건너뛰고 W&B가 각 파일을 기본 루트나 지정된 다운로드 디렉토리에 다운로드합니다. |
|  `path_prefix` |  지정된 경우, 주어진 접두사로 시작하는 경로만 다운로드됩니다. 유닉스 형식(슬래시 사용)을 사용합니다. |

| 반환값 |  |
| :--- | :--- |
|  다운로드된 콘텐츠의 경로입니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |
|  `RuntimeError` |  오프라인 모드에서 아티팩트를 다운로드하려고 하면. |

### `file`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1994-L2019)

```python
file(
    root: Optional[str] = None
) -> StrPath
```

단일 파일 아티팩트를 `root`로 지정한 디렉토리에 다운로드합니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  파일을 저장할 루트 디렉토리입니다. 기본값은 './artifacts/self.name/'입니다. |

| 반환값 |  |
| :--- | :--- |
|  다운로드된 파일의 전체 경로입니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |
|  `ValueError` |  아티팩트에 여러 파일이 포함된 경우. |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L2021-L2038)

```python
files(
    names: Optional[List[str]] = None,
    per_page: int = 50
) -> ArtifactFiles
```

이 아티팩트에 저장된 모든 파일을 반복합니다.

| 인수 |  |
| :--- | :--- |
|  `names` |  나열하려는 아티팩트의 루트를 기준으로 한 파일 경로입니다. |
|  `per_page` |  요청당 반환할 파일 수입니다. |

| 반환값 |  |
| :--- | :--- |
|  `File` 오브젝트를 포함한 반복자. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |

### `finalize`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L740-L748)

```python
finalize() -> None
```

아티팩트 버전을 확정합니다.

아티팩트 버전이 확정된 후에는 더 이상 수정할 수 없습니다. 아티팩트를 특정 아티팩트 버전으로 로그했기 때문입니다. 새로운 아티팩트 버전을 생성하여 더 많은 데이터를 아티팩트로 로그하세요. `log_artifact`로 아티팩트를 로그하면 아티팩트가 자동으로 확정됩니다.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1590-L1636)

```python
get(
    name: str
) -> Optional[data_types.WBValue]
```

아티팩트 상대 `name`에 위치한 WBValue 오브젝트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  검색할 아티팩트 상대 이름입니다. |

| 반환값 |  |
| :--- | :--- |
|  `wandb.log()`를 사용해 로그할 수 있고 W&B UI에서 시각화할 수 있는 W&B 오브젝트입니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않았거나 run이 오프라인인 경우 |

### `get_added_local_path_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1638-L1650)

```python
get_added_local_path_name(
    local_path: str
) -> Optional[str]
```

로컬 파일 시스템 경로로 추가된 파일의 아티팩트 상대 이름을 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `local_path` |  아티팩트 상대 이름으로 해석할 로컬 경로입니다. |

| 반환값 |  |
| :--- | :--- |
|  아티팩트 상대 이름. |

### `get_entry`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1568-L1588)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

주어진 이름의 엔트리를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  가져올 아티팩트 상대 이름입니다. |

| 반환값 |  |
| :--- | :--- |
|  `W&amp;B` 오브젝트. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않았거나 run이 오프라인인 경우. |
|  `KeyError` |  주어진 이름의 엔트리가 아티팩트에 포함되지 않은 경우. |

### `get_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1560-L1566)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

Deprecated. `get_entry(name)`을 사용하세요.

### `is_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L758-L763)

```python
is_draft() -> bool
```

아티팩트가 저장되지 않았는지 확인합니다.

반환값: Boolean. 아티팩트가 저장된 경우 `False`. 아티팩트가 저장되지 않은 경우 `True`.

### `json_encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L2273-L2280)

```python
json_encode() -> Dict[str, Any]
```

아티팩트를 JSON 형식으로 인코딩하여 반환합니다.

| 반환값 |  |
| :--- | :--- |
|  아티팩트의 속성을 나타내는 `string` 키를 가진 `dict`. |

### `link`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L2109-L2137)

```python
link(
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

이 아티팩트를 포트폴리오에 연결합니다 (아티팩트의 승격된 컬렉션).

| 인수 |  |
| :--- | :--- |
|  `target_path` |  프로젝트 내 포트폴리오의 경로입니다. 대상 경로는 `{portfolio}`, `{project}/{portfolio}` 또는 `{entity}/{project}/{portfolio}` 스키마 중 하나를 준수해야 합니다. 프로젝트 내 일반 포트폴리오가 아닌 Model Registry에 아티팩트를 연결하려면, `target_path`를 `{"model-registry"}/{Registered Model Name}` 또는 `{entity}/{"model-registry"}/{Registered Model Name}` 스키마로 설정하세요. |
|  `aliases` |  지정된 포트폴리오 내에 아티팩트를 고유하게 식별하는 문자열 목록입니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |

### `logged_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L2228-L2271)

```python
logged_by() -> Optional[Run]
```

원래 아티팩트를 로그한 W&B run을 가져옵니다.

| 반환값 |  |
| :--- | :--- |
|  원래 아티팩트를 로그한 W&B run의 이름입니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |

### `new_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L355-L387)

```python
new_draft() -> "Artifact"
```

이 확정된 아티팩트와 동일한 내용의 새 초안 아티팩트를 생성합니다.

반환된 아티팩트는 확장하거나 수정할 수 있으며 새 버전으로 로그할 수 있습니다.

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |

### `new_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1167-L1204)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "w",
    encoding: Optional[str] = None
) -> Generator[IO, None, None]
```

새 임시 파일을 열고 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  아티팩트에 추가할 새 파일의 이름입니다. |
|  `mode` |  새 파일을 열기 위해 사용할 파일 엑세스 모드입니다. |
|  `encoding` |  새 파일을 열 때 사용하는 인코딩입니다. |

| 반환값 |  |
| :--- | :--- |
|  작성 가능한 새 파일 오브젝트입니다. 닫으면 자동으로 아티팩트에 추가됩니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 이미 확정되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |

### `remove`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1529-L1558)

```python
remove(
    item: Union[StrPath, 'ArtifactManifestEntry']
) -> None
```

아티팩트에서 항목을 제거합니다.

| 인수 |  |
| :--- | :--- |
|  `item` |  제거할 항목입니다. 특정 매니페스트 항목 또는 아티팩트 상대 경로의 이름일 수 있습니다. 항목이 디렉토리와 일치하면 해당 디렉토리의 모든 항목이 제거됩니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 이미 확정되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |
|  `FileNotFoundError` |  아티팩트에서 항목을 찾을 수 없는 경우. |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L768-L807)

```python
save(
    project: Optional[str] = None,
    settings: Optional['wandb.sdk.wandb_settings.Settings'] = None
) -> None
```

아티팩트에 대한 모든 변경 사항을 저장합니다.

현재 run 중이라면, run이 이 아티팩트를 로그합니다. 현재 run 중이 아니면, 이 아티팩트를 추적하기 위한 "auto" 유형의 run이 생성됩니다.

| 인수 |  |
| :--- | :--- |
|  `project` |  현재 로그가 진행 중이 아닌 경우 아티팩트를 사용할 프로젝트입니다. |
|  `settings` |  자동 run을 초기화할 때 사용할 설정 오브젝트입니다. 주로 테스트 환경에서 사용됩니다. |

### `unlink`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L2139-L2155)

```python
unlink() -> None
```

이 아티팩트가 포트폴리오(승격된 아티팩트 컬렉션)의 멤버인 경우 이 아티팩트의 연결을 해제합니다.

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |
|  `ValueError` |  아티팩트가 링크되지 않았거나 포트폴리오 컬렉션의 멤버가 아닌 경우. |

### `used_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L2181-L2226)

```python
used_by() -> List[Run]
```

이 아티팩트를 사용한 run의 목록을 가져옵니다.

| 반환값 |  |
| :--- | :--- |
|  `Run` 오브젝트의 목록. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |

### `verify`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1953-L1992)

```python
verify(
    root: Optional[str] = None
) -> None
```

아티팩트의 내용이 매니페스트와 일치하는지 확인합니다.

디렉토리 내 모든 파일의 체크섬을 계산하고, 체크섬을 아티팩트의 매니페스트와 교차 참조합니다. 참조는 확인되지 않습니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  확인할 디렉토리입니다. None인 경우 아티팩트는 './artifacts/self.name/'에 다운로드됩니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우. |
|  `ValueError` |  검증이 실패한 경우. |

### `wait`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L815-L836)

```python
wait(
    timeout: Optional[int] = None
) -> "Artifact"
```

필요한 경우 이 아티팩트가 로그를 완료할 때까지 대기합니다.

| 인수 |  |
| :--- | :--- |
|  `timeout` |  대기할 시간(초)입니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1137-L1149)

```python
__getitem__(
    name: str
) -> Optional[data_types.WBValue]
```

아티팩트 상대 `name`에 위치한 WBValue 오브젝트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  가져올 아티팩트 상대 이름입니다. |

| 반환값 |  |
| :--- | :--- |
|  `wandb.log()`를 사용해 로그할 수 있고 W&B UI에서 시각화할 수 있는 W&B 오브젝트입니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않았거나 run이 오프라인인 경우. |

### `__setitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/sdk/artifacts/artifact.py#L1151-L1165)

```python
__setitem__(
    name: str,
    item: data_types.WBValue
) -> ArtifactManifestEntry
```

`item`을 아티팩트의 `name` 경로에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  오브젝트를 추가할 아티팩트 내 경로입니다. |
|  `item` |  추가할 오브젝트입니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목 |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 이미 확정되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |