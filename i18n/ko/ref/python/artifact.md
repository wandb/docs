
# Artifact

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L80-L2231' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


데이터셋 및 모델 버전 관리를 위한 유연하고 가벼운 구성 요소.

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

빈 W&B 아티팩트를 생성합니다. `add`로 시작하는 메소드를 사용하여 아티팩트의 내용을 채웁니다. 원하는 모든 파일을 아티팩트에 추가한 후, `wandb.log_artifact()`를 호출하여 로그합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  아티팩트의 인간이 읽을 수 있는 이름입니다. W&B App UI 또는 프로그래밍 방식으로 특정 아티팩트를 식별하는 데 이름을 사용합니다. `use_artifact` 공개 API로 아티팩트를 대화식으로 참조할 수 있습니다. 이름에는 문자, 숫자, 밑줄, 하이픈, 점이 포함될 수 있습니다. 이름은 프로젝트 전체에서 고유해야 합니다. |
|  `type` |  아티팩트의 유형입니다. 아티팩트의 유형을 사용하여 아티팩트를 조직하고 구별합니다. 문자, 숫자, 밑줄, 하이픈, 점이 포함된 문자열을 사용할 수 있습니다. 일반적인 유형에는 `dataset` 또는 `model`이 포함됩니다. 아티팩트를 W&B 모델 레지스트리에 연결하려면 유형 문자열 내에 `model`을 포함시키세요. |
|  `description` |  아티팩트의 설명입니다. 모델 또는 데이터셋 아티팩트의 경우, 표준화된 팀 모델 또는 데이터셋 카드에 대한 문서를 추가합니다. `Artifact.description` 속성이나 W&B App UI로 프로그래밍 방식으로 아티팩트의 설명을 봅니다. W&B는 W&B App에서 설명을 마크다운으로 렌더링합니다. |
|  `metadata` |  아티팩트에 대한 추가 정보입니다. 메타데이터를 키-값 쌍의 사전으로 지정합니다. 최대 100개의 키를 지정할 수 있습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 객체입니다. |

| 속성 |  |
| :--- | :--- |
|  `aliases` |  아티팩트 버전에 할당된 하나 이상의 의미론적으로 친숙한 참조 또는 "별명" 목록입니다. 별명은 프로그래밍 방식으로 참조할 수 있는 변경 가능한 참조입니다. W&B App UI 또는 프로그래밍 방식으로 아티팩트의 별명을 변경하세요. 자세한 정보는 [새 아티팩트 버전 생성](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version)을 참조하세요. |
|  `collection` |  이 아티팩트가 검색된 컬렉션입니다. 컬렉션은 아티팩트 버전의 정렬된 그룹입니다. 아티팩트가 포트폴리오/연결된 컬렉션에서 검색된 경우, 아티팩트 버전이 원래 나온 컬렉션 대신 해당 컬렉션이 반환됩니다. 아티팩트가 원래 나온 컬렉션을 소스 시퀀스라고 합니다. |
|  `commit_hash` |  이 아티팩트가 커밋되었을 때 반환된 해시입니다. |
|  `created_at` |  아티팩트가 생성된 타임스탬프입니다. |
|  `description` |  아티팩트의 설명입니다. |
|  `digest` |  아티팩트의 논리적 다이제스트입니다. 다이제스트는 아티팩트 내용의 체크섬입니다. 아티팩트가 현재 `latest` 버전과 같은 다이제스트를 가지고 있다면, `log_artifact`는 수행되지 않습니다. |
|  `entity` |  이차(포트폴리오) 아티팩트 컬렉션의 엔티티 이름입니다. |
|  `file_count` |  파일(참조 포함)의 수입니다. |
|  `id` |  아티팩트의 ID입니다. |
|  `manifest` |  아티팩트의 매니페스트입니다. 매니페스트는 모든 내용을 나열하며, 아티팩트가 로그된 후에는 변경할 수 없습니다. |
|  `metadata` |  사용자 정의 아티팩트 메타데이터입니다. 아티팩트와 관련된 구조화된 데이터입니다. |
|  `name` |  이차(포트폴리오) 컬렉션에서의 아티팩트 이름과 버전입니다. {collection}:{alias} 형식의 문자열입니다. 아티팩트가 저장되기 전에는 버전이 아직 알려지지 않았기 때문에 이름만 포함됩니다. |
|  `project` |  이차(포트폴리오) 아티팩트 컬렉션의 프로젝트 이름입니다. |
|  `qualified_name` |  이차(포트폴리오) 컬렉션의 entity/project/name입니다. |
|  `size` |  아티팩트의 총 크기(바이트)입니다. 이 아티팩트에서 추적하는 모든 참조가 포함됩니다. |
|  `source_collection` |  아티팩트의 기본(시퀀스) 컬렉션입니다. |
|  `source_entity` |  기본(시퀀스) 아티팩트 컬렉션의 엔티티 이름입니다. |
|  `source_name` |  기본(시퀀스) 컬렉션에서의 아티팩트 이름과 버전입니다. {collection}:{alias} 형식의 문자열입니다. 아티팩트가 저장되기 전에는 버전이 아직 알려지지 않았기 때문에 이름만 포함됩니다. |
|  `source_project` |  기본(시퀀스) 아티팩트 컬렉션의 프로젝트 이름입니다. |
|  `source_qualified_name` |  기본(시퀀스) 컬렉션의 entity/project/name입니다. |
|  `source_version` |  기본(시퀀스) 컬렉션에서의 아티팩트 버전입니다. "v{number}" 형식의 문자열입니다. |
|  `state` |  아티팩트의 상태입니다. "PENDING", "COMMITTED", 또는 "DELETED" 중 하나입니다. |
|  `ttl` |  아티팩트의 수명(TTL) 정책입니다. TTL 정책의 지속 기간이 지나면 아티팩트가 곧 삭제됩니다. `None`으로 설정하면, 아티팩트는 TTL 정책을 비활성화하고, 팀 기본 TTL이 있더라도 삭제되지 않도록 예약됩니다. 아티팩트는 팀 관리자가 기본 TTL을 정의하고 아티팩트에 사용자 지정 정책이 설정되지 않은 경우 팀 기본값에서 TTL 정책을 상속받습니다. |
|  `type` |  아티팩트의 유형입니다. 일반적인 유형에는 `dataset` 또는 `model`이 포함됩니다. |
|  `updated_at` |  아티팩트가 마지막으로 업데이트된 시간입니다. |
|  `version` |  이차(포트폴리오) 컬렉션에서의 아티팩트 버전입니다. |

## 메소드

### `add`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1289-L1386)

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
|  `name` |  아티팩트 내에 객체를 추가할 경로입니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목입니다. |

| 발생하는 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 최종화되었기 때문에 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |

### `add_dir`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1164-L1207)

```python
add_dir(
    local_path: str,
    name: Optional[str] = None
) -> None
```

로컬 디렉토리를 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `local_path` |  로컬 디렉토리의 경로입니다. |
|  `name` |  아티팩트 내의 하위 디렉토리 이름입니다. 지정한 이름은 W&B App UI에서 아티팩트의 `type`으로 중첩되어 표시됩니다. 기본값은 아티팩트의 루트입니다. |

| 발생하는 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 최종화되었기 때문에 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |

### `add_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1127-L1162)

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
|  `name` |  아티팩트 내에서 사용될 파일의 경로입니다. 기본값은 파일의 기본 이름입니다. |
|  `is_tmp` |  참이면, 파일은 충돌을 피하기 위해 결정적으로 이름이 변경됩니다. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목입니다. |

| 발생하는 오류 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전은 최종화되었기 때문에 변경할 수 없습니다. 대신 새 아티팩트 버전을 로그하세요. |

### `add_reference`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1209-L1287)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

URI로 표시된 참조를 아티팩트에 추가합니다.

아티팩트에 추가하는 파일이나 디렉토리와 달리, 참조는 W&B에 업로드되지 않습니다. 자세한 정보는 [외부 파일 추적](https://docs.wandb.ai/guides/artifacts/track-external-files)을 참조하세요.

기본적으로 다음 스키마가 지원됩니다:

- http(s): 파일의 크기와 다이제스트는 서버가 반환한 `Content-Length` 및 `ETag` 응답 헤더에 의해 추론됩니다.
- s3: 체크섬과 크기는 객체 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우 버전 ID도 추적됩니다.
- gs: 체크섬과 크기는 객체 메타데이터에서 가져옵니다. 버킷 버전 관리가 활성화된 경우 버전 ID도 추적됩니다.
- https, `*.blob.core.windows.net` 도메인과 일치하는 경우(Azure): 체크섬과 크기는 블롭 메타데이터에서 가져옵니다. 저장소 계정 버전 관리가 활성화된 경우 버전 ID도 추적됩니다.
- file: 체크섬과 크기는 파일 시스템에서 가져옵니다. NFS 공유 또는 추적하고자 하는 파일을 포함하는 다른 외부에 마운트된 볼륨이 있지만 반드시 업로드하고 싶지 않은 경우 이 스키마가 유용합니다.

다른 스키마의 경우, 다이제스트는 단순히 URI의 해시이며 크기는 비워 둡니다.

| 인수 |  |
| :--- | :--- |
|  `uri` |  추가할 참조의 URI 경로입니다. `Artifact.get_entry`에서 반환된 객체를 사용하여 다른 아티팩트 항목에 대한 참조를 저장할 수 있습니다. |
|  `name` |  이 참조의 내용을 아티팩트 내에 배치할 경로입니다. |
|  `checksum` |  참조 URI에서 위치한 리소스에 대한 체크섬을 계산할지 여부입니다. 체크섬은 자동 무결성 검증을 가능하게 하므로 강력히 권장되지만, 아

### `new_draft`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L324-L356)

```python
new_draft() -> "Artifact"
```

이 커밋된 아티팩트와 동일한 내용으로 새로운 드래프트 아티팩트를 생성합니다.

반환된 아티팩트는 확장되거나 수정되어 새 버전으로 기록될 수 있습니다.

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 기록되지 않았을 경우. |

### `new_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1088-L1125)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "w",
    encoding: Optional[str] = None
) -> Generator[IO, None, None]
```

새로운 임시 파일을 열고 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  아티팩트에 추가할 새 파일의 이름. |
|  `mode` |  새 파일을 열기 위해 사용할 파일 엑세스 모드. |
|  `encoding` |  새 파일을 열 때 사용할 인코딩. |

| 반환값 |  |
| :--- | :--- |
|  작성할 수 있는 새 파일 오브젝트. 닫히면 파일이 자동으로 아티팩트에 추가됩니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전이 완료되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 기록하세요. |

### `path_contains_dir_prefix`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1594-L1601)

```python
@classmethod
path_contains_dir_prefix(
    path: StrPath,
    dir_path: StrPath
) -> bool
```

`path`가 `dir_path`를 접두사로 포함하는 경우 true를 반환합니다.

### `remove`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1407-L1436)

```python
remove(
    item: Union[StrPath, 'ArtifactManifestEntry']
) -> None
```

아티팩트에서 항목을 제거합니다.

| 인수 |  |
| :--- | :--- |
|  `item` |  제거할 항목. 특정 매니페스트 항목이나 아티팩트 상대 경로의 이름일 수 있습니다. 항목이 디렉토리와 일치하는 경우 해당 디렉토리의 모든 항목이 제거됩니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전이 완료되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 기록하세요. |
|  `FileNotFoundError` |  아티팩트에서 항목을 찾을 수 없는 경우. |

### `save`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L730-L769)

```python
save(
    project: Optional[str] = None,
    settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

아티팩트에 대한 변경 사항을 유지합니다.

현재 run이 진행 중인 경우, 해당 run은 이 아티팩트를 기록합니다. 현재 run이 없는 경우, 이 아티팩트를 추적하기 위해 "auto" 유형의 run이 생성됩니다.

| 인수 |  |
| :--- | :--- |
|  `project` |  run이 이미 컨텍스트에 없는 경우 아티팩트를 사용할 프로젝트. |
|  `settings` |  자동 run을 초기화할 때 사용할 설정 오브젝트. 주로 테스트 하네스에서 사용됩니다. |

### `should_download_entry`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1603-L1609)

```python
@classmethod
should_download_entry(
    entry: ArtifactManifestEntry,
    prefix: Optional[StrPath]
) -> bool
```

### `used_by`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L2012-L2057)

```python
used_by() -> List[Run]
```

이 아티팩트를 사용한 run의 목록을 가져옵니다.

| 반환값 |  |
| :--- | :--- |
|  `Run` 오브젝트의 목록. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 기록되지 않았을 경우. |

### `verify`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1833-L1872)

```python
verify(
    root: Optional[str] = None
) -> None
```

아티팩트의 내용이 매니페스트와 일치하는지 확인합니다.

디렉토리의 모든 파일이 체크섬되고, 이 체크섬들은 아티팩트의 매니페스트와 대조됩니다. 참조는 검증되지 않습니다.

| 인수 |  |
| :--- | :--- |
|  `root` |  검증할 디렉토리. None인 경우 아티팩트는 './artifacts/self.name/'에 다운로드됩니다. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 기록되지 않았을 경우. |
|  `ValueError` |  검증에 실패한 경우. |

### `wait`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L777-L798)

```python
wait(
    timeout: Optional[int] = None
) -> "Artifact"
```

필요한 경우, 이 아티팩트가 기록을 완료할 때까지 기다립니다.

| 인수 |  |
| :--- | :--- |
|  `timeout` |  기다릴 시간(초). |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트. |

### `__getitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1058-L1070)

```python
__getitem__(
    name: str
) -> Optional[data_types.WBValue]
```

아티팩트 상대 `name`에 위치한 WBValue 오브젝트를 가져옵니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  가져올 아티팩트 상대 이름. |

| 반환값 |  |
| :--- | :--- |
|  `wandb.log()`로 기록되고 W&B UI에서 시각화될 수 있는 W&B 오브젝트. |

| 예외 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  아티팩트가 기록되지 않았거나 run이 오프라인인 경우. |

### `__setitem__`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/sdk/artifacts/artifact.py#L1072-L1086)

```python
__setitem__(
    name: str,
    item: data_types.WBValue
) -> ArtifactManifestEntry
```

`name` 경로에 `item`을 아티팩트에 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  오브젝트를 추가할 아티팩트 내 경로. |
|  `item` |  추가할 오브젝트. |

| 반환값 |  |
| :--- | :--- |
|  추가된 매니페스트 항목 |

| 예외 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  현재 아티팩트 버전이 완료되어 변경할 수 없습니다. 대신 새 아티팩트 버전을 기록하세요. |