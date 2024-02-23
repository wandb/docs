
# 아티팩트

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L80-L2237' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

데이터세트와 모델 버전 관리를 위한 유연하고 가벼운 빌딩 블록입니다.

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

빈 W&B 아티팩트를 구성합니다. `add`로 시작하는 메서드를 사용하여 아티팩트의 내용을 채웁니다. 아티팩트에 원하는 모든 파일이 포함되면, `wandb.log_artifact()`를 호출하여 기록할 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  아티팩트에 대한 사람이 읽을 수 있는 이름입니다. 이름을 사용하여 W&B 앱 UI 또는 프로그래밍 방식으로 특정 아티팩트를 식별합니다. `use_artifact` 공개 API로 아티팩트를 대화형으로 참조할 수 있습니다. 이름에는 글자, 숫자, 밑줄, 하이픈 및 점이 포함될 수 있습니다. 이름은 프로젝트 전체에서 고유해야 합니다. |
|  `type` |  아티팩트의 타입입니다. 아티팩트의 타입을 사용하여 아티팩트를 구성하고 구별합니다. 글자, 숫자, 밑줄, 하이픈 및 점이 포함된 문자열을 사용할 수 있습니다. 일반적인 타입에는 `dataset` 또는 `model`이 포함됩니다. 아티팩트를 W&B 모델 레지스트리에 연결하려면 타입 문자열 내에 `model`을 포함합니다. |
|  `description` |  아티팩트에 대한 설명입니다. 모델 또는 데이터세트 아티팩트의 경우, 표준화된 팀 모델 또는 데이터세트 카드에 대한 문서를 추가합니다. W&B 앱 UI 또는 프로그래밍 방식으로 아티팩트의 설명을 `Artifact.description` 속성으로 확인할 수 있습니다. W&B는 W&B 앱에서 마크다운으로 설명을 렌더링합니다. |
|  `metadata` |  아티팩트에 대한 추가 정보입니다. 메타데이터를 키-값 쌍의 사전으로 지정합니다. 최대 100개의 키를 지정할 수 있습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 객체입니다. |

| 속성 |  |
| :--- | :--- |
|  `aliases` |  아티팩트 버전에 할당된 하나 이상의 의미 친화적 참조 또는 "별칭" 목록입니다. 별칭은 프로그래밍 방식으로 참조할 수 있는 변경 가능한 참조입니다. W&B 앱 UI 또는 프로그래밍 방식으로 아티팩트의 별칭을 변경할 수 있습니다. 자세한 내용은 [새로운 아티팩트 버전 생성](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version)을 참조하십시오. |
|  `collection` |  이 아티팩트가 검색된 컬렉션입니다. 컬렉션은 아티팩트 버전의 순서 있는 그룹입니다. 이 아티팩트가 포트폴리오/연결된 컬렉션에서 검색된 경우, 아티팩트 버전이 원래 생성된 컬렉션이 아니라 해당 컬렉션이 반환됩니다. 아티팩트가 원래 생성된 컬렉션을 원천 시퀀스라고 합니다. |
|  `commit_hash` |  이 아티팩트가 커밋될 때 반환된 해시입니다. |
|  `created_at` |  아티팩트가 생성된 시간의 타임스탬프입니다. |
|  `description` |  아티팩트에 대한 설명입니다. |
|  `digest` |  아티팩트의 논리적 다이제스트입니다. 다이제스트는 아티팩트 내용의 체크섬입니다. 아티팩트가 현재 `latest` 버전과 동일한 다이제스트를 가지고 있다면, `log_artifact`는 아무 작업도 수행하지 않습니다. |
|  `entity` |  보조(포트폴리오) 아티팩트 컬렉션의 엔티티 이름입니다. |
|  `file_count` |  파일(참조 포함)의 수입니다. |
|  `id` |  아티팩트의 ID입니다. |
|  `manifest` |  아티팩트의 매니페스트입니다. 매니페스트는 아티팩트의 모든 내용을 나열하며, 아티팩트가 기록된 후에는 변경할 수 없습니다. |
|  `metadata` |  사용자 정의 아티팩트 메타데이터입니다. 아티팩트와 관련된 구조화된 데이터입니다. |
|  `name` |  보조(포트폴리오) 컬렉션에서의 아티팩트 이름과 버전입니다. {collection}:{alias} 형식의 문자열입니다. 아티팩트가 저장되기 전에는 버전이 아직 알려지지 않았으므로 이름만 포함됩니다. |
|  `project` |  보조(포트폴리오) 아티팩트 컬렉션의 프로젝트 이름입니다. |
|  `qualified_name` |  보조(포트폴리오) 컬렉션의 entity/project/name입니다. |
|  `size` |  바이트 단위로 측정된 아티팩트의 총 크기입니다. 이 아티팩트에서 추적하는 모든 참조를 포함합니다. |
|  `source_collection` |  아티팩트의 기본(시퀀스) 컬렉션입니다. |
|  `source_entity` |  기본(시퀀스) 아티팩트 컬렉션의 엔티티 이름입니다. |
|  `source_name` |  기본(시퀀스) 컬렉션에서의 아티팩트 이름과 버전입니다. {collection}:{alias} 형식의 문자열입니다. 아티팩트가 저장되기 전에는 버전이 아직 알려지지 않았으므로 이름만 포함됩니다. |
|  `source_project` |  기본(시퀀스) 아티팩트 컬렉션의 프로젝트 이름입니다. |
|  `source_qualified_name` |  기본(시퀀스) 컬렉션의 entity/project/name입니다. |
|  `source_version` |  기본(시퀀스) 컬렉션에서의 아티팩트 버전입니다. "v{number}" 형식의 문자열입니다. |
|  `state` |  아티팩트의 상태입니다. "PENDING", "COMMITTED", 또는 "DELETED" 중 하나입니다. |
|  `ttl` |  아티팩트의 생존 시간(TTL) 정책입니다. TTL 정책의 기간이 지나면 아티팩트는 곧 삭제됩니다. `None`으로 설정된 경우, 아티팩트는 TTL 정책을 비활성화하고 팀 기본 TTL이 있더라도 삭제 예정으로 설정되지 않습니다. 아티팩트는 팀 관리자가 기본 TTL을 정의하고 아티팩트에 사용자 지정 정책이 설정되지 않은 경우 팀 기본값에서 TTL 정책을 상속받습니다. |
|  `type` |  아티팩트의 타입입니다. 일반적인 타입에는 `dataset` 또는 `model`이 포함됩니다. |
|  `updated_at` |  아티팩트가 마지막으로 업데이트된 시간입니다. |
|  `version` |  보조(포트폴리오) 컬렉션에서의 아티팩트 버전입니다. |

## 메서드

### `add`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1295-L1392)

```python
add(
    obj: data_types.WBValue,
    name: StrPath
) -> ArtifactManifestEntry
```

wandb.WBValue `obj`를 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `obj` |  추가할 객체입니다. 현재 Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D 중 하나를 지원합니다. |
|  `name` |  아티팩트 내에서 객체를 추가할 경로입니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 최종화되었기 때문에 변경할 수 없습니다. 대신 새로운 아티팩트 버전을 기록하세요. |

### `add_dir`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1170-L1213)

```python
add_dir(
    local_path: str,
    name: Optional[str] = None
) -> None
```

로컬 디렉터리를 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `local_path` |  로컬 디렉터리의 경로입니다. |
|  `name` |  아티팩트 내의 서브디렉터리 이름입니다. 지정한 이름은 아티팩트의 `type`에 따라 W&B 앱 UI에 중첩되어 나타납니다. 기본값은 아티팩트의 루트입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 최종화되었기 때문에 변경할 수 없습니다. 대신 새로운 아티팩트 버전을 기록하세요. |

### `add_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1133-L1168)

```python
add_file(
    local_path: str,
    name: Optional[str] = None,
    is_tmp: Optional[bool] = (False)
) -> ArtifactManifestEntry
```

로컬 파일을 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `local_path` |  추가되는 파일의 경로입니다. |
|  `name` |  추가되는 파일을 위해 아티팩트 내에서 사용할 경로입니다. 파일의 기본 이름을 기본값으로 합니다. |
|  `is_tmp` |  참일 경우, 파일은 충돌을 피하기 위해 결정론적으로 이름이 변경됩니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 최종화되었기 때문에 변경할 수 없습니다. 대신 새로운 아티팩트 버전을 기록하세요. |

### `add_reference`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1215-L1293)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

URI로 표시된 참조를 아티팩트에 추가합니다.

아티팩트에 추가하는 파일이나 디렉터리와 달리, 참조는 W&B에 업로드되지 않습니다. 자세한 내용은 [외부 파일 추적](https://docs.wandb.ai/guides/artifacts/track-external-files)을 참조하십시오.

기본적으로 다음 스키마가 지원됩니다:

- http(s): 서버가 반환하는 `Content-Length` 및 `ETag` 응답 헤더에 의해 파일의 크기와 다이제스트가 추론됩니다.
- s3: 체크섬과 크기는 오브젝트 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우, 버전 ID도 추적됩니다.
- gs: 체크섬과 크기는 오브젝트 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우, 버전 ID도 추적됩니다.
- https, `*.blob.core.windows.net` 도메인 일치(Azure): 체크섬과 크기는 블롭 메타데이터에서 가져옵니다. 스토리지 계정 버전 관리가 활성화된 경우, 버전 ID도 추적됩니다.
- file: 체크섬과 크기는 파일 시스템에서 가져옵니다. 이 스키마는 추적하고 싶지만 반드시 업로드하지는 않아야 하는 NFS 공유나 다른 외부로 마운트된 볼륨에 있는 파일이 있는 경우 유용합니다.

다른 스키마의 경우, 다이제스트는 URI의 해시이며 크기는 비워둡니다.

| 인수 |  |
| :--- | :--- |
|  `uri` |  추가할 참조의 URI 경로입니다. `

### `finalize`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L702-L710)

```python
finalize() -> None
```

아티팩트 버전을 확정합니다.

아티팩트가 특정 버전으로 로그되면 그 아티팩트 버전을 수정할 수 없습니다. 아티팩트에 데이터를 더 로그하려면 새 아티팩트 버전을 생성하세요. `log_artifact`로 아티팩트를 로그할 때 아티팩트는 자동으로 확정됩니다.

### `get`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1473-L1519)

```python
get(
    name: str
) -> Optional[data_types.WBValue]
```

아티팩트 상대 `name`에 위치한 WBValue 개체를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  가져올 아티팩트 상대 이름입니다. |

| 반환값 |  |
| :--- | :--- |
|  `wandb.log()`로 로그할 수 있고 W&B UI에서 시각화할 수 있는 W&B 개체입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않았거나 실행이 오프라인 상태인 경우 |

### `get_added_local_path_name`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1521-L1533)

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
|  아티팩트 상대 이름입니다. |

### `get_entry`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1451-L1471)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

주어진 이름을 가진 엔트리를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  가져올 아티팩트 상대 이름입니다. |

| 반환값 |  |
| :--- | :--- |
|  `W&amp;B` 개체입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않았거나 실행이 오프라인 상태인 경우입니다. |
|  `KeyError` |  주어진 이름을 가진 엔트리가 아티팩트에 없는 경우입니다. |

### `get_path`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1444-L1449)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

사용되지 않습니다. `get_entry(name)`을 사용하세요.

### `is_draft`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L720-L725)

```python
is_draft() -> bool
```

아티팩트가 저장되지 않았는지 확인합니다.

반환값: 아티팩트가 저장된 경우 `False`, 저장되지 않은 경우 `True`인 Boolean입니다.

### `json_encode`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L2110-L2117)

```python
json_encode() -> Dict[str, Any]
```

아티팩트를 JSON 형식으로 인코딩하여 반환합니다.

| 반환값 |  |
| :--- | :--- |
|  아티팩트의 속성을 나타내는 `string` 키를 가진 `dict`입니다. |

### `link`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1988-L2016)

```python
link(
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

이 아티팩트를 포트폴리오(승진된 아티팩트 모음)에 연결합니다.

| 인수 |  |
| :--- | :--- |
|  `target_path` |  프로젝트 내의 포트폴리오로의 경로입니다. 대상 경로는 다음 스키마 중 하나를 준수해야 합니다. `{portfolio}`, `{project}/{portfolio}` 또는 `{entity}/{project}/{portfolio}`. 아티팩트를 프로젝트 내의 일반 포트폴리오가 아닌 모델 레지스트리에 연결하려면 `target_path`를 다음 스키마로 설정하세요. `{"model-registry"}/{Registered Model Name}` 또는 `{entity}/{"model-registry"}/{Registered Model Name}`. |
|  `aliases` |  지정된 포트폴리오 내에서 아티팩트를 고유하게 식별하는 문자열 목록입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우입니다. |

### `logged_by`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L2065-L2108)

```python
logged_by() -> Optional[Run]
```

이 아티팩트를 처음 로그한 W&B 실행을 가져옵니다.

| 반환값 |  |
| :--- | :--- |
|  이 아티팩트를 처음 로그한 W&B 실행의 이름입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우입니다. |

### `new_draft`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L324-L356)

```python
new_draft() -> "Artifact"
```

이 커밋된 아티팩트와 동일한 내용으로 새 드래프트 아티팩트를 생성합니다.

반환된 아티팩트는 확장되거나 수정되어 새 버전으로 로그될 수 있습니다.

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 개체입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우입니다. |

### `new_file`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1094-L1131)

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
|  `mode` |  새 파일을 열 때 사용할 파일 접근 모드입니다. |
|  `encoding` |  새 파일을 열 때 사용할 인코딩입니다. |

| 반환값 |  |
| :--- | :--- |
|  쓸 수 있는 새 파일 개체입니다. 닫을 때 파일은 자동으로 아티팩트에 추가됩니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 확정되었으므로 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |

### `path_contains_dir_prefix`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1600-L1607)

```python
@classmethod
path_contains_dir_prefix(
    path: StrPath,
    dir_path: StrPath
) -> bool
```

`path`가 `dir_path`를 접두사로 포함하는지 여부를 반환합니다.

### `remove`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1413-L1442)

```python
remove(
    item: Union[StrPath, 'ArtifactManifestEntry']
) -> None
```

아티팩트에서 항목을 제거합니다.

| 인수 |  |
| :--- | :--- |
|  `item` |  제거할 항목입니다. 구체적인 매니페스트 엔트리 또는 아티팩트 상대 경로의 이름일 수 있습니다. 항목이 디렉터리와 일치하면 해당 디렉터리의 모든 항목이 제거됩니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 확정되었으므로 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |
|  `FileNotFoundError` |  항목이 아티팩트에서 찾을 수 없는 경우입니다. |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L730-L769)

```python
save(
    project: Optional[str] = None,
    settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

아티팩트에 대한 변경 사항을 영구적으로 저장합니다.

현재 실행 중인 경우, 해당 실행이 이 아티팩트를 로그합니다. 현재 실행 중이 아닌 경우, "auto" 유형의 실행이 이 아티팩트를 추적하도록 생성됩니다.

| 인수 |  |
| :--- | :--- |
|  `project` |  실행이 이미 컨텍스트에 없는 경우 사용할 아티팩트의 프로젝트입니다. |
|  `settings` |  자동 실행을 초기화할 때 사용할 설정 개체입니다. 주로 테스트 하네스에서 사용됩니다. |

### `should_download_entry`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1609-L1615)

```python
@classmethod
should_download_entry(
    entry: ArtifactManifestEntry,
    prefix: Optional[StrPath]
) -> bool
```

### `used_by`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L2018-L2063)

```python
used_by() -> List[Run]
```

이 아티팩트를 사용한 실행 목록을 가져옵니다.

| 반환값 |  |
| :--- | :--- |
|  `Run` 개체의 목록입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우입니다. |

### `verify`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1839-L1878)

```python
verify(
    root: Optional[str] = None
) -> None
```

아티팩트의 내용이 매니페스트와 일치하는지 확인합니다.

디렉터리의 모든 파일은 체크섬되며, 그 체크섬은 아티팩트의 매니페스트와 교차 참조됩니다. 참조는 검증되지 않습니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  검증할 디렉터리입니다. None인 경우 아티팩트는 './artifacts/self.name/'에 다운로드됩니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않은 경우입니다. |
|  `ValueError` |  검증이 실패한 경우입니다. |

### `wait`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L777-L804)

```python
wait(
    timeout: Optional[int] = None
) -> "Artifact"
```

필요한 경우 이 아티팩트가 로그를 마칠 때까지 기다립니다.

| 인수 |  |
| :--- | :--- |
|  `timeout` |  기다릴 시간(초)입니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 개체입니다. |

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1064-L1076)

```python
__getitem__(
    name: str
) -> Optional[data_types.WBValue]
```

아티팩트 상대 `name`에 위치한 WBValue 개체를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  가져올 아티팩트 상대 이름입니다. |

| 반환값 |  |
| :--- | :--- |
|  `wandb.log()`로 로그할 수 있고 W&B UI에서 시각화할 수 있는 W&B 개체입니다. |

| 발생 가능한 오류 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 로그되지 않았거나 실행이 오프라인 상태인 경우입니다. |

### `__setitem__`

[View source](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/sdk/artifacts/artifact.py#L1078-L1092)

```python
__setitem__(
    name: str,
    item: data_types.WBValue
) -> ArtifactManifestEntry
```

`name` 경로에 `item`을 아티팩트에 추가합니다.

