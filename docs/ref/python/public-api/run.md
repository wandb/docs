# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L272-L901' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

하나의 Run은 Entity와 Project와 연결됩니다.

```python
Run(
    client: "RetryingClient",
    entity: str,
    project: str,
    run_id: str,
    attrs: Optional[Mapping] = None,
    include_sweeps: bool = (True)
)
```

| 속성 |  |
| :--- | :--- |

## Methods

### `create`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L371-L411)

```python
@classmethod
create(
    api, run_id=None, project=None, entity=None
)
```

주어진 프로젝트에 대한 Run을 만듭니다.

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L529-L557)

```python
delete(
    delete_artifacts=(False)
)
```

wandb 백엔드에서 주어진 Run을 삭제합니다.

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 오브젝트를 주피터에서 표시합니다.

### `file`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L619-L629)

```python
file(
    name
)
```

아티팩트에서 주어진 이름의 파일 경로를 반환합니다.

| 인수 |  |
| :--- | :--- |
|  name (str): 요청된 파일의 이름입니다. |

| 반환값 |  |
| :--- | :--- |
|  이름 인수와 일치하는 `File`입니다. |

### `files`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L606-L617)

```python
files(
    names=None, per_page=50
)
```

이름이 지정된 각 파일의 파일 경로를 반환합니다.

| 인수 |  |
| :--- | :--- |
|  names (list): 요청된 파일의 이름입니다. 비어 있으면 모든 파일을 반환합니다. per_page (int): 페이지당 결과 수입니다. |

| 반환값 |  |
| :--- | :--- |
|  `File` 오브젝트의 반복자 역할을 하는 `Files` 오브젝트입니다. |

### `history`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L655-L695)

```python
history(
    samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

Run의 샘플링된 히스토리 메트릭을 반환합니다.

히스토리 기록이 샘플링되는 것을 수용할 수 있다면 이 방법이 더 간단하고 빠릅니다.

| 인수 |  |
| :--- | :--- |
|  `samples` |  (int, optional) 반환할 샘플 수 |
|  `pandas` |  (bool, optional) 판다스 데이터프레임을 반환합니다 |
|  `keys` |  (list, optional) 특정 키에 대한 메트릭만 반환합니다 |
|  `x_axis` |  (str, optional) 기본값은 _step입니다. 이 메트릭을 xAxis로 사용합니다 |
|  `stream` |  (str, optional) 메트릭용 "default", 시스템 메트릭용 "system" |

| 반환값 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  pandas=True인 경우 히스토리 메트릭의 `pandas.DataFrame`을 반환합니다. pandas=False인 경우 히스토리 메트릭의 dict의 목록을 반환합니다. |

### `load`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L413-L477)

```python
load(
    force=(False)
)
```

### `log_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L794-L828)

```python
log_artifact(
    artifact, aliases=None
)
```

아티팩트를 Run의 출력으로 선언합니다.

| 인수 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`에서 반환된 아티팩트 aliases (list, optional): 이 아티팩트에 적용할 에일리어스 |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

### `logged_artifacts`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L753-L755)

```python
logged_artifacts(
    per_page=100
)
```

### `save`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L559-L560)

```python
save()
```

### `scan_history`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L697-L751)

```python
scan_history(
    keys=None, page_size=1000, min_step=None, max_step=None
)
```

Run에 대한 모든 히스토리 기록의 반복 가능한 컬렉션을 반환합니다.

#### 예시:

예시 Run의 모든 손실 값을 내보내기

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

| 인수 |  |
| :--- | :--- |
|  keys ([str], optional): 이러한 키만 가져오고, 모든 키가 정의된 행만 가져옵니다. page_size (int, optional): api에서 가져올 페이지의 크기입니다. min_step (int, optional): 한 번에 스캔할 페이지의 최소 수입니다. max_step (int, optional): 한 번에 스캔할 페이지의 최대 수입니다. |

| 반환값 |  |
| :--- | :--- |
|  히스토리 기록(dict)에 대한 반복 가능한 컬렉션입니다. |

### `snake_to_camel`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L887-L895)

```python
to_html(
    height=420, hidden=(False)
)
```

이 Run을 표시하는 iframe을 포함한 HTML을 생성합니다.

### `update`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L502-L527)

```python
update()
```

wandb 백엔드에 Run 오브젝트의 변경 사항을 저장합니다.

### `upload_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L631-L653)

```python
upload_file(
    path, root="."
)
```

파일을 업로드합니다.

| 인수 |  |
| :--- | :--- |
|  path (str): 업로드할 파일의 이름입니다. root (str): 파일을 상대적으로 저장할 기본 경로입니다. 예를 들어 현재 "my_dir"에 있고, run 내에서 "my_dir/file.txt"로 파일을 저장하려면 root를 "../"로 설정해야 합니다. |

| 반환값 |  |
| :--- | :--- |
|  이름 인수와 일치하는 `File`입니다. |

### `use_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L761-L792)

```python
use_artifact(
    artifact, use_as=None
)
```

아티팩트를 Run의 입력으로 선언합니다.

| 인수 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`에서 반환된 아티팩트 use_as (string, optional): 스크립트에서 아티팩트가 사용되는 방법을 식별하는 문자열입니다. 베타 wandb launch 기능의 아티팩트 스와핑 기능을 사용하여 run에서 사용되는 아티팩트를 쉽게 구분할 수 있습니다. |

| 반환값 |  |
| :--- | :--- |
|  `Artifact` 오브젝트입니다. |

### `used_artifacts`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L757-L759)

```python
used_artifacts(
    per_page=100
)
```

### `wait_until_finished`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/apis/public/runs.py#L479-L500)

```python
wait_until_finished()
```