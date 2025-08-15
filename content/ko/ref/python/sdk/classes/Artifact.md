---
title: 아티팩트
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-classes-Artifact
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/artifacts/artifact.py >}}




## <kbd>class</kbd> `Artifact`
데이터셋 및 모델 버전 관리를 위한 유연하고 가벼운 빌딩 블록입니다.

비어 있는 W&B Artifact를 생성합니다. 아티팩트의 내용은 `add`로 시작하는 메소드를 통해 추가할 수 있습니다. 원하는 파일을 모두 추가한 뒤 `run.log_artifact()`를 호출하여 로그를 남길 수 있습니다.



**파라미터:**

 - `name` (str):  사람이 읽기 쉬운 아티팩트 이름입니다. 이 이름을 사용하여 W&B App UI 또는 코드에서 특정 아티팩트를 식별할 수 있습니다. `use_artifact` Public API를 통해 인터랙티브하게 참조할 수 있습니다. 이름에는 영문자, 숫자, 언더스코어, 하이픈, 점을 사용할 수 있습니다. 프로젝트 내에서 고유해야 합니다.
 - `type` (str):  아티팩트의 타입입니다. 타입을 사용하여 아티팩트를 구성하거나 구분할 수 있습니다. 영문자, 숫자, 언더스코어, 하이픈, 점을 포함한 임의의 문자열을 사용할 수 있습니다. 자주 사용되는 타입으로는 `dataset` 또는 `model`이 있습니다. W&B Model Registry와 연결하려면 타입 문자열에 `model`을 포함하세요. 내부 사용을 위해 예약된 타입(`job` 등 및 `wandb-`로 시작하는 타입)은 직접 설정할 수 없습니다.
 - `description (str | None) = None`:  아티팩트에 대한 설명입니다. Model 또는 Dataset Artifacts의 경우, 팀 모델이나 데이터셋 카드를 문서화할 수 있습니다. `Artifact.description` 속성을 통해 프로그래밍적으로 확인하거나, W&B App UI에서 확인할 수 있습니다. W&B App에서는 마크다운 형식으로 렌더링됩니다.
 - `metadata (dict[str, Any] | None) = None`:  아티팩트에 대한 추가 정보입니다. 키-값 쌍으로 이루어진 사전을 지정합니다. 최대 100개의 키까지만 설정할 수 있습니다.
 - `incremental`:  기존 아티팩트를 수정하려면 `Artifact.new_draft()` 메소드를 사용하세요.
 - `use_as`:  사용 중단됨.
 - `is_link`:  아티팩트가 연결된 아티팩트(`True`)인지, 원본 아티팩트(`False`)인지를 나타내는 불리언 값입니다.



**반환값:**
 `Artifact` 오브젝트가 반환됩니다.

### <kbd>method</kbd> `Artifact.__init__`

```python
__init__(
    name: 'str',
    type: 'str',
    description: 'str | None' = None,
    metadata: 'dict[str, Any] | None' = None,
    incremental: 'bool' = False,
    use_as: 'str | None' = None
) → None
```

---

### <kbd>property</kbd> Artifact.aliases

아티팩트 버전에 할당된 의미 있는 별칭 또는 식별자 목록입니다.

별칭은 프로그래밍적으로 참조할 수 있으며, W&B App UI 또는 프로그래밍 방식으로 변경할 수 있습니다. 자세한 내용은 [새 아티팩트 버전 생성](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) 을 참고하세요.

---

### <kbd>property</kbd> Artifact.collection

이 아티팩트가 가져온 컬렉션입니다.

컬렉션은 아티팩트 버전의 순서 있는 그룹입니다. 아티팩트가 포트폴리오/연결된 컬렉션에서 가져온 경우, 원본 컬렉션이 아닌 해당 컬렉션이 반환됩니다. 아티팩트가 처음 생성된 컬렉션을 "source sequence"라고 합니다.

---

### <kbd>property</kbd> Artifact.commit_hash

이 아티팩트가 커밋되었을 때 반환되는 해시 값입니다.

---

### <kbd>property</kbd> Artifact.created_at

아티팩트가 생성된 시점을 나타내는 타임스탬프입니다.

---

### <kbd>property</kbd> Artifact.description

아티팩트에 대한 설명입니다.

---

### <kbd>property</kbd> Artifact.digest

아티팩트의 논리적 다이제스트(체크섬)입니다.

다이제스트는 아티팩트의 내용으로부터 계산된 체크섬입니다. 아티팩트의 다이제스트가 현재 `latest` 버전과 동일하다면, `log_artifact`는 아무 작업도 하지 않습니다.

---

### <kbd>property</kbd> Artifact.entity

아티팩트 컬렉션이 속한 Entity 이름입니다.

아티팩트가 링크인 경우, 연결된 아티팩트의 Entity가 반환됩니다.

---

### <kbd>property</kbd> Artifact.file_count

파일(참조 포함)의 개수입니다.

---

### <kbd>property</kbd> Artifact.history_step

아티팩트 원본 run의 히스토리 메트릭이 로깅된 가장 가까운 step입니다.

**예시:**
```python
run = artifact.logged_by()
if run and (artifact.history_step is not None):
     history = run.sample_history(
         min_step=artifact.history_step,
         max_step=artifact.history_step + 1,
         keys=["my_metric"],
     )
```

---

### <kbd>property</kbd> Artifact.id

아티팩트의 ID 입니다.

---

### <kbd>property</kbd> Artifact.is_link

아티팩트가 링크 아티팩트인지 여부를 나타내는 플래그입니다.

True: 이 아티팩트는 소스 아티팩트에 대한 링크입니다.  
False: 이 아티팩트는 소스 아티팩트입니다.

---

### <kbd>property</kbd> Artifact.linked_artifacts

소스 아티팩트의 모든 링크된 아티팩트의 리스트를 반환합니다.

만약 이 아티팩트가 링크 아티팩트(`artifact.is_link == True`)라면, 빈 리스트를 반환합니다. 최대 500개까지 표시됩니다.

---

### <kbd>property</kbd> Artifact.manifest

아티팩트의 매니페스트입니다.

매니페스트에는 모든 내용 목록이 포함되며, 아티팩트가 로그되면 더 이상 변경할 수 없습니다.

---

### <kbd>property</kbd> Artifact.metadata

사용자 정의 아티팩트 메타데이터입니다.

아티팩트와 연결된 구조화된 데이터입니다.

---

### <kbd>property</kbd> Artifact.name

아티팩트의 이름과 버전입니다.

`{collection}:{alias}` 형식의 문자열입니다. 아티팩트가 로그/저장되기 전에 가져오면 alias가 포함되지 않을 수 있습니다. 아티팩트가 링크인 경우, 연결된 아티팩트의 이름으로 표시됩니다.

---

### <kbd>property</kbd> Artifact.project

아티팩트 컬렉션이 속한 프로젝트의 이름입니다.

아티팩트가 링크라면, 연결된 아티팩트가 속한 프로젝트 이름이 반환됩니다.

---

### <kbd>property</kbd> Artifact.qualified_name

아티팩트의 entity/project/name 정보입니다.

아티팩트가 링크라면 연결 아티팩트 경로의 qualified name이 반환됩니다.

---

### <kbd>property</kbd> Artifact.size

아티팩트의 전체 크기(바이트 단위)입니다.

이 아티팩트에서 추적하는 모든 참조를 포함합니다.

---

### <kbd>property</kbd> Artifact.source_artifact

원본 아티팩트를 반환합니다. 원본 아티팩트란, 최초로 로그된 아티팩트입니다.

아티팩트 자체가 소스 아티팩트(`artifact.is_link == False`)라면 자신을 반환합니다.

---

### <kbd>property</kbd> Artifact.source_collection

아티팩트의 소스 컬렉션입니다.

소스 컬렉션은 아티팩트가 로그된 컬렉션을 의미합니다.

---

### <kbd>property</kbd> Artifact.source_entity

원본 아티팩트의 Entity 명입니다.

---

### <kbd>property</kbd> Artifact.source_name

원본 아티팩트의 이름과 버전입니다.

`{source_collection}:{alias}` 형태의 문자열입니다. 아티팩트가 저장되기 전에는 이름만 포함되고 버전은 알 수 없습니다.

---

### <kbd>property</kbd> Artifact.source_project

원본 아티팩트가 소속된 프로젝트 이름입니다.

---

### <kbd>property</kbd> Artifact.source_qualified_name

원본 아티팩트의 entity/project/name 정보입니다.

---

### <kbd>property</kbd> Artifact.source_version

원본 아티팩트의 버전입니다.

`v{number}` 형식의 문자열입니다.

---

### <kbd>property</kbd> Artifact.state

아티팩트의 상태입니다. 다음 중 하나: "PENDING", "COMMITTED", "DELETED".

---

### <kbd>property</kbd> Artifact.tags

이 아티팩트 버전에 할당된 태그 목록입니다.

---

### <kbd>property</kbd> Artifact.ttl

아티팩트의 TTL(Time-To-Live) 정책입니다.

TTL 정책이 지난 후, Artifacts는 곧 삭제됩니다. `None`으로 설정하면 TTL 정책이 비활성화되어 팀 기본 TTL이 있더라도 삭제가 예약되지 않습니다. 별도 정책이 없고 팀 관리자가 기본 TTL을 정의했다면, 팀 기본값을 상속받습니다.

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그 또는 저장되지 않았다면, 상속된 TTL을 가져올 수 없습니다.

---

### <kbd>property</kbd> Artifact.type

아티팩트의 타입입니다. 주로 사용되는 타입은 `dataset` 또는 `model`입니다.

---

### <kbd>property</kbd> Artifact.updated_at

아티팩트가 마지막으로 업데이트된 시각입니다.

---

### <kbd>property</kbd> Artifact.url

아티팩트의 URL을 생성합니다.

**반환값:**
 
 - `str`:  아티팩트의 URL

---

### <kbd>property</kbd> Artifact.use_as

사용 중단됨.

---

### <kbd>property</kbd> Artifact.version

아티팩트의 버전입니다.

`v{number}` 형식의 문자열입니다. 만약 이 아티팩트가 링크 아티팩트라면, 링크된 컬렉션의 버전이 표시됩니다.

---

### <kbd>method</kbd> `Artifact.add`

```python
add(
    obj: 'WBValue',
    name: 'StrPath',
    overwrite: 'bool' = False
) → ArtifactManifestEntry
```

wandb.WBValue `obj`를 아티팩트에 추가합니다.

**파라미터:**
 
 - `obj`:  추가할 오브젝트입니다. 현재 지원: Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D,  Audio, Image, Video, Html, Object3D
 - `name`:  오브젝트를 추가할 아티팩트 내 경로입니다.
 - `overwrite`:  True이면, 동일한 경로에 이미 객체가 있는 경우 기존 객체를 덮어씁니다.

**반환값:**
 추가된 매니페스트 엔트리

**예외:**
 
 - `ArtifactFinalizedError`:  현재 아티팩트 버전이 최종화되어 더 이상 변경할 수 없습니다. 새 아티팩트 버전으로 로그하세요.

---

### <kbd>method</kbd> `Artifact.add_dir`

```python
add_dir(
    local_path: 'str',
    name: 'str | None' = None,
    skip_cache: 'bool | None' = False,
    policy: "Literal['mutable', 'immutable'] | None" = 'mutable',
    merge: 'bool' = False
) → None
```

로컬 디렉토리를 아티팩트에 추가합니다.

**파라미터:**
 
 - `local_path`:  추가할 로컬 디렉토리 경로입니다.
 - `name`:  아티팩트 내에 생성할 서브디렉토리 이름입니다. 이 이름은 W&B App UI에서 artifact의 `type`별로 구분되어 나타납니다. 기본값은 아티팩트의 루트입니다.
 - `skip_cache`:  True로 설정하면 업로드 중 파일을 캐시에 복사/이동하지 않습니다.
 - `policy`:  기본값 "mutable".
    - mutable: 업로드 중 손상 방지를 위해 임시로 복사본 생성
    - immutable: 보호를 비활성화하고, 사용자가 파일을 삭제/수정하지 않을 것이라 가정
 - `merge`:  False(기본값)이면 이전 add_dir에서 추가한 파일이 변경됐을 경우 ValueError 발생, True 이면 파일이 변경되면 기존 파일을 덮어씁니다. 항상 새 파일은 추가하며 삭제는 되지 않습니다. 전체 디렉토리를 대체하려면 name을 지정해 추가 후 `remove(my_prefix)`로 해당 디렉토리를 삭제한 뒤 추가하세요.

**예외:**
 
 - `ArtifactFinalizedError`:  현재 아티팩트 버전이 최종화되어 더 이상 변경할 수 없습니다. 새 아티팩트 버전으로 로그하세요.
 - `ValueError`:  policy는 "mutable" 혹은 "immutable"만 허용됩니다.

---

### <kbd>method</kbd> `Artifact.add_file`

```python
add_file(
    local_path: 'str',
    name: 'str | None' = None,
    is_tmp: 'bool | None' = False,
    skip_cache: 'bool | None' = False,
    policy: "Literal['mutable', 'immutable'] | None" = 'mutable',
    overwrite: 'bool' = False
) → ArtifactManifestEntry
```

로컬 파일을 아티팩트에 추가합니다.

**파라미터:**
 
 - `local_path`:  추가할 파일 경로입니다.
 - `name`:  아티팩트 내에서 사용할 파일 경로. 기본값은 파일의 basename입니다.
 - `is_tmp`:  True이면, 충돌 방지를 위해 파일 이름이 결정적으로 변경됩니다.
 - `skip_cache`:  True로 설정하면 업로드 후 파일을 캐시에 복사하지 않습니다.
 - `policy`:  "mutable"(기본값)이면 파일 업로드 시 임시 파일 복사본 생성, "immutable"이면 복사하지 않고 사용자가 파일을 변경/삭제하지 않는다고 간주합니다.
 - `overwrite`:  True이면 동일한 이름의 파일이 있으면 덮어씁니다.

**반환값:**
 추가된 매니페스트 엔트리

**예외:**
 
 - `ArtifactFinalizedError`:  현재 아티팩트 버전이 최종화되어 더 이상 변경할 수 없습니다. 새 아티팩트 버전으로 로그하세요.
 - `ValueError`:  policy는 "mutable" 혹은 "immutable"만 허용됩니다.

---

### <kbd>method</kbd> `Artifact.add_reference`

```python
add_reference(
    uri: 'ArtifactManifestEntry | str',
    name: 'StrPath | None' = None,
    checksum: 'bool' = True,
    max_objects: 'int | None' = None
) → Sequence[ArtifactManifestEntry]
```

URI로 표시된 참조를 아티팩트에 추가합니다.

파일이나 디렉토리처럼 업로드하지 않고, 외부 리소스를 참조만 합니다. 자세한 내용은 [외부 파일 추적하기](https://docs.wandb.ai/guides/artifacts/track-external-files)를 확인하세요.

다음 스킴이 기본적으로 지원됩니다:

- http(s): 서버가 반환하는 `Content-Length` 와 `ETag` 헤더를 기반으로 파일 크기 및 다이제스트 결정
- s3: 오브젝트 메타데이터에서 체크섬과 크기를 가져옴. 버킷 버전 관리가 설정된 경우 버전 ID도 추적
- gs: 위와 동등, 버킷 버전 관리 지원 시 버전 ID 추적
- https 및 `*.blob.core.windows.net` 도메인: 
- Azure: blob 메타데이터로 체크섬과 크기, 스토리지 계정 버전 관리 시 버전 ID 추적
- file: 파일 시스템에서 체크섬과 크기 확인 — 예: NFS 드라이브 등

기타 스킴은 URI의 해시를 다이제스트로 사용하며, 크기는 비워둡니다.

**파라미터:**
 
 - `uri`:  추가할 참조의 URI 경로. 다른 아티팩트의 엔트리를 참조하려면, `Artifact.get_entry`에서 반환한 오브젝트도 지정 가능
 - `name`:  이 참조를 아티팩트 내에 저장할 경로
 - `checksum`:  참조 URI에 위치한 리소스의 체크섬 사용 여부. 무결성 검증을 위해 권장하지만, 비활성화하면 속도는 빨라집니다. 디렉토리 참조시 체크섬 비활성화시 하위 오브젝트는 저장하지 않고, 참조 URI 변경 시에만 새 버전이 생성됩니다.
 - `max_objects`:  디렉토리나 버킷 프리픽스 등 다수 오브젝트 참조 시, 최대 가져올 오브젝트 개수. S3, GCS, Azure, 로컬 파일은 기본 1,000만개까지 허용. 그 외는 제한 없음.

**반환값:**
 추가된 매니페스트 엔트리 목록

**예외:**
 
 - `ArtifactFinalizedError`:  현재 아티팩트 버전이 최종화되어 더 이상 변경할 수 없습니다. 새 아티팩트 버전으로 로그하세요.

---

### <kbd>method</kbd> `Artifact.checkout`

```python
checkout(root: 'str | None' = None) → str
```

지정한 루트 디렉토리를 아티팩트의 내용으로 교체합니다.

주의: 아티팩트에 포함되지 않은 `root` 내 모든 파일은 삭제됩니다.

**파라미터:**
 
 - `root`:  이 아티팩트의 파일로 대체할 디렉토리

**반환값:**
 체크아웃된 내용의 경로

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

---

### <kbd>method</kbd> `Artifact.delete`

```python
delete(delete_aliases: 'bool' = False) → None
```

아티팩트 및 파일 삭제

링크된 아티팩트라면 링크만 삭제되고 원본은 유지됩니다.

소스-링크 간 연결을 삭제하려면 `artifact.delete()` 대신 `artifact.unlink()`를 사용하세요.

**파라미터:**
 
 - `delete_aliases`:  True로 설정하면 이 아티팩트와 연결된 모든 별칭을 삭제합니다.  그렇지 않으면 별칭이 존재할 경우 예외가 발생합니다. 링크된(포트폴리오 멤버인) 경우 무시됨.

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

---

### <kbd>method</kbd> `Artifact.download`

```python
download(
    root: 'StrPath | None' = None,
    allow_missing_references: 'bool' = False,
    skip_cache: 'bool | None' = None,
    path_prefix: 'StrPath | None' = None,
    multipart: 'bool | None' = None
) → FilePathStr
```

아티팩트의 내용을 지정한 루트 디렉토리에 다운로드합니다.

기존 `root` 안의 파일은 수정되지 않습니다. 아티팩트와 동일하게 맞추려면 다운로드 전에 `root`를 직접 삭제하세요.

**파라미터:**
 
 - `root`:  아티팩트 파일을 저장할 디렉토리
 - `allow_missing_references`:  True이면 유효하지 않은 참조 경로는 무시되고 다운로드를 계속합니다.
 - `skip_cache`:  True이면 artifact cache를 사용하지 않고, W&B가 각 파일을 지정한 디렉토리로 직접 다운로드합니다.
 - `path_prefix`:  지정하면, 해당 프리픽스로 시작하는 경로만 다운로드합니다. unix 포맷(슬래시) 사용
 - `multipart`:  None(기본값)이면, 파일 크기가 2GB 초과 시 멀티파트로 병렬 다운로드합니다. True/False로 지정하면 파일 크기에 상관없이 병렬/직렬 다운로드를 적용합니다.

**반환값:**
 다운로드된 콘텐츠의 경로

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

---

### <kbd>method</kbd> `Artifact.file`

```python
file(root: 'str | None' = None) → StrPath
```

단일 파일 아티팩트를 지정한 `root` 디렉토리에 다운로드합니다.

**파라미터:**
 
 - `root`:  파일을 저장할 루트 디렉토리. 기본값은 `./artifacts/self.name/`입니다.

**반환값:**
 다운로드된 전체 파일 경로

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우
 - `ValueError`:  아티팩트에 파일이 하나보다 많으면 예외 발생

---

### <kbd>method</kbd> `Artifact.files`

```python
files(names: 'list[str] | None' = None, per_page: 'int' = 50) → ArtifactFiles
```

이 아티팩트에 저장된 모든 파일을 순회합니다.

**파라미터:**
 
 - `names`:  아티팩트 내에서 나열하고 싶은 파일 이름(루트 기준)
 - `per_page`:  요청당 반환할 파일 수

**반환값:**
 `File` 오브젝트를 포함하는 이터레이터

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

---

### <kbd>method</kbd> `Artifact.finalize`

```python
finalize() → None
```

아티팩트 버전을 최종화(finalize)합니다.

최종화된 이후에는 더 이상 해당 버전을 수정할 수 없습니다. 추가 데이터를 저장하려면 새 아티팩트 버전을 만들어야 합니다. 아티팩트는 `log_artifact`로 로그할 때 자동으로 최종화됩니다.

---

### <kbd>method</kbd> `Artifact.get`

```python
get(name: 'str') → WBValue | None
```

아티팩트 상대 `name`에 위치한 WBValue 오브젝트를 가져옵니다.

**파라미터:**
 
 - `name`:  가져올 아티팩트 내 상대 경로

**반환값:**
 `run.log()`로 로그하고 W&B UI에서 시각화할 수 있는 W&B 오브젝트

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않았거나 Run이 오프라인인 경우

---

### <kbd>method</kbd> `Artifact.get_added_local_path_name`

```python
get_added_local_path_name(local_path: 'str') → str | None
```

로컬 파일 시스템 경로로 추가한 파일의 아티팩트 상대 경로를 가져옵니다.

**파라미터:**
 
 - `local_path`:  아티팩트 상대 경로로 변환할 로컬 경로

**반환값:**
 아티팩트 내 상대 경로

---

### <kbd>method</kbd> `Artifact.get_entry`

```python
get_entry(name: 'StrPath') → ArtifactManifestEntry
```

지정한 이름을 가진 엔트리를 가져옵니다.

**파라미터:**
 
 - `name`:  가져올 아티팩트 상대 경로

**반환값:**
 `W&B` 오브젝트

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않았거나 run이 오프라인이면
 - `KeyError`:  해당 이름의 엔트리가 아티팩트에 없으면

---

### <kbd>method</kbd> `Artifact.get_path`

```python
get_path(name: 'StrPath') → ArtifactManifestEntry
```

사용 중단됨. 대신 `get_entry(name)`을 사용하세요.

---

### <kbd>method</kbd> `Artifact.is_draft`

```python
is_draft() → bool
```

아티팩트가 아직 저장되지 않았는지 확인합니다.

**반환값:**
  불리언. 아티팩트가 저장됐다면 `False`, 저장 전이면 `True`.

---

### <kbd>method</kbd> `Artifact.json_encode`

```python
json_encode() → dict[str, Any]
```

아티팩트를 JSON 포맷으로 인코딩해서 반환합니다.

**반환값:**
  아티팩트 속성을 나타내는 `string` 키의 `dict`

---

### <kbd>method</kbd> `Artifact.link`

```python
link(target_path: 'str', aliases: 'list[str] | None' = None) → Artifact | None
```

이 아티팩트를 포트폴리오(승격된 아티팩트 컬렉션)에 링크합니다.

**파라미터:**
 
 - `target_path`:  프로젝트 내 포트폴리오 경로. `{portfolio}`, `{project}/{portfolio}`, `{entity}/{project}/{portfolio}` 스키마 중 하나여야 합니다. Model Registry에 등록하려면 `{"model-registry"}/{Registered Model Name}` 혹은 `{entity}/{"model-registry"}/{Registered Model Name}` 스키마로 설정합니다.
 - `aliases`:  포트폴리오 내에서 아티팩트를 고유하게 식별할 수 있는 별칭(문자열) 목록

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

**반환값:**
 링크에 성공했다면 링크된 아티팩트, 실패 시 None

---

### <kbd>method</kbd> `Artifact.logged_by`

```python
logged_by() → Run | None
```

이 아티팩트를 최초로 로그한 W&B run을 가져옵니다.

**반환값:**
  아티팩트를 최초로 로그한 W&B run의 이름

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

---

### <kbd>method</kbd> `Artifact.new_draft`

```python
new_draft() → Artifact
```

이 커밋된 아티팩트와 동일한 내용을 가진 새 드래프트 아티팩트를 만듭니다.

기존 아티팩트 수정은 "증분 아티팩트(incremental artifact)"라는 새 버전 생성으로 처리되며, 반환된 아티팩트는 수정 및 확장 후 새로운 버전으로 로그할 수 있습니다.

**반환값:**
  `Artifact` 오브젝트

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

---

### <kbd>method</kbd> `Artifact.new_file`

```python
new_file(
    name: 'str',
    mode: 'str' = 'x',
    encoding: 'str | None' = None
) → Iterator[IO]
```

새 임시 파일을 열어서 아티팩트에 추가합니다.

**파라미터:**
 
 - `name`:  아티팩트에 추가할 새 파일의 이름
 - `mode`:  새 파일을 열 때 사용할 파일 엑세스 모드
 - `encoding`:  새 파일을 열 때 사용할 인코딩

**반환값:**
 쓰기가 가능한 새 파일 오브젝트. 파일을 닫으면 자동으로 아티팩트에 추가됨

**예외:**
 
 - `ArtifactFinalizedError`:  현재 아티팩트 버전이 최종화되어 더 이상 변경할 수 없습니다. 새 아티팩트 버전으로 로그하세요.

---

### <kbd>method</kbd> `Artifact.remove`

```python
remove(item: 'StrPath | ArtifactManifestEntry') → None
```

아티팩트에서 항목을 제거합니다.

**파라미터:**
 
 - `item`:  제거할 항목(특정 매니페스트 엔트리 또는 아티팩트 내 상대 경로). 디렉토리 경로라면 해당 디렉토리 내 모든 항목을 삭제

**예외:**
 
 - `ArtifactFinalizedError`:  현재 아티팩트 버전이 최종화되어 더 이상 변경할 수 없습니다. 새 아티팩트 버전으로 로그하세요.
 - `FileNotFoundError`:  해당 항목을 아티팩트에서 찾을 수 없는 경우

---

### <kbd>method</kbd> `Artifact.save`

```python
save(
    project: 'str | None' = None,
    settings: 'wandb.Settings | None' = None
) → None
```

아티팩트에 가한 변경 사항을 영구 저장합니다.

현재 run에서 실행 중이면 해당 run이 이 아티팩트를 로그합니다. 그렇지 않은 경우 "auto" 타입의 run을 자동으로 생성하여 추적합니다.

**파라미터:**
 
 - `project`:  run 컨텍스트가 없을 때 사용할 프로젝트 이름
 - `settings`:  자동 run을 초기화할 때 사용할 wandb.Settings 오브젝트 (테스트용 등)

---

### <kbd>method</kbd> `Artifact.unlink`

```python
unlink() → None
```

이 아티팩트가 승격된 아티팩트 컬렉션(포트폴리오)의 멤버라면 연결을 해제합니다.

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우
 - `ValueError`:  아티팩트가 링크되어 있지 않은(포트폴리오 멤버가 아닌) 경우

---

### <kbd>method</kbd> `Artifact.used_by`

```python
used_by() → list[Run]
```

이 아티팩트 및 링크된 아티팩트를 사용한 run의 목록을 가져옵니다.

**반환값:**
  `Run` 오브젝트 리스트

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우

---

### <kbd>method</kbd> `Artifact.verify`

```python
verify(root: 'str | None' = None) → None
```

아티팩트 내용이 매니페스트와 일치하는지 검증합니다.

디렉토리 내 모든 파일을 체크섬하여, 아티팩트 매니페스트와 비교합니다. 참조(Reference)는 검증하지 않습니다.

**파라미터:**
 
 - `root`:  검증할 디렉토리. None이면 './artifacts/self.name/'로 다운로드 후 검증

**예외:**
 
 - `ArtifactNotLoggedError`:  아티팩트가 로그되지 않은 경우
 - `ValueError`:  검증에 실패한 경우

---

### <kbd>method</kbd> `Artifact.wait`

```python
wait(timeout: 'int | None' = None) → Artifact
```

필요한 경우 이 아티팩트의 로깅이 끝날 때까지 대기합니다.

**파라미터:**
 
 - `timeout`:  대기할 시간(초 단위)

**반환값:**
 `Artifact` 오브젝트