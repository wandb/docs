
# 실행

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L168-L802' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

개체와 프로젝트에 연결된 단일 실행입니다.

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

## 메서드

### `create`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L267-L307)

```python
@classmethod
create(
    api, run_id=None, project=None, entity=None
)
```

주어진 프로젝트를 위한 실행을 생성합니다.

### `delete`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L427-L455)

```python
delete(
    delete_artifacts=(False)
)
```

wandb 백엔드에서 주어진 실행을 삭제합니다.

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

이 개체를 jupyter에 표시합니다.

### `file`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L518-L528)

```python
file(
    name
)
```

아티팩트에서 주어진 이름의 파일 경로를 반환합니다.

| 인수 |  |
| :--- | :--- |
|  name (str): 요청된 파일의 이름. |

| 반환 |  |
| :--- | :--- |
|  이름 인수와 일치하는 `File`. |

### `files`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L505-L516)

```python
files(
    names=None, per_page=50
)
```

이름이 주어진 각 파일에 대한 파일 경로를 반환합니다.

| 인수 |  |
| :--- | :--- |
|  names (list): 요청된 파일의 이름, 비어 있으면 모든 파일을 페이지당 결과 수로 반환합니다(per_page (int): 페이지 당 결과 수). |

| 반환 |  |
| :--- | :--- |
|  `Files` 개체, `File` 개체들에 대한 반복자입니다. |

### `history`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L554-L594)

```python
history(
    samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

실행에 대한 샘플링된 기록 메트릭을 반환합니다.

기록 레코드가 샘플링되어도 괜찮다면 더 간단하고 빠릅니다.

| 인수 |  |
| :--- | :--- |
|  `samples` |  (int, 선택사항) 반환할 샘플의 수 |
|  `pandas` |  (bool, 선택사항) pandas 데이터프레임을 반환 |
|  `keys` |  (list, 선택사항) 특정 키에 대한 메트릭만 반환 |
|  `x_axis` |  (str, 선택사항) 이 메트릭을 xAxis로 사용 기본값은 _step |
|  `stream` |  (str, 선택사항) 메트릭에는 "default", 기계 메트릭에는 "system" |

| 반환 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  pandas=True이면 기록 메트릭의 `pandas.DataFrame`을 반환합니다. 리스트 of dicts: pandas=False이면 기록 메트릭의 dict 리스트를 반환합니다. |

### `load`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L309-L374)

```python
load(
    force=(False)
)
```

### `log_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L692-L729)

```python
log_artifact(
    artifact, aliases=None
)
```

실행의 출력으로 아티팩트를 선언합니다.

| 인수 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`에서 반환된 아티팩트 aliases (list, 선택사항): 이 아티팩트에 적용할 별칭 |

| 반환 |  |
| :--- | :--- |
|  `Artifact` 개체. |

### `logged_artifacts`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L651-L653)

```python
logged_artifacts(
    per_page=100
)
```

### `save`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L457-L458)

```python
save()
```

### `scan_history`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L596-L649)

```python
scan_history(
    keys=None, page_size=1000, min_step=None, max_step=None
)
```

실행에 대한 모든 기록 레코드의 반복 가능한 컬렉션을 반환합니다.

#### 예시:

예시 실행에 대한 모든 손실 값 내보내기

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

| 인수 |  |
| :--- | :--- |
|  keys ([str], 선택사항): 이 키들만 가져오고, 모든 키가 정의된 행만 가져옵니다. page_size (int, 선택사항): api에서 가져올 페이지 크기 |

| 반환 |  |
| :--- | :--- |
|  기록 레코드(dict)에 대한 반복 가능한 컬렉션. |

### `snake_to_camel`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L788-L796)

```python
to_html(
    height=420, hidden=(False)
)
```

이 실행을 표시하는 iframe이 포함된 HTML 생성합니다.

### `update`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L399-L425)

```python
update()
```

wandb 백엔드에 실행 개체에 대한 변경 사항을 반영합니다.

### `upload_file`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L530-L552)

```python
upload_file(
    path, root="."
)
```

파일을 업로드합니다.

| 인수 |  |
| :--- | :--- |
|  path (str): 업로드할 파일의 이름. root (str): 파일을 상대적으로 저장할 루트 경로. 즉, 파일을 실행에서 "my_dir/file.txt"로 저장하려면 현재 "my_dir"에 있다면 root를 "../"로 설정합니다. |

| 반환 |  |
| :--- | :--- |
|  이름 인수와 일치하는 `File`. |

### `use_artifact`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L659-L690)

```python
use_artifact(
    artifact, use_as=None
)
```

실행의 입력으로 아티팩트를 선언합니다.

| 인수 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`에서 반환된 아티팩트 use_as (string, 선택사항): 스크립트에서 아티팩트를 사용하는 방식을 식별하는 문자열. 베타 wandb 실행 기능의 아티팩트 교환 기능을 사용할 때 실행에서 사용되는 아티팩트를 쉽게 구별하는 데 사용됩니다. |

| 반환 |  |
| :--- | :--- |
|  `Artifact` 개체. |

### `used_artifacts`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L655-L657)

```python
used_artifacts(
    per_page=100
)
```

### `wait_until_finished`

[소스 보기](https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/apis/public/runs.py#L376-L397)

```python
wait_until_finished()
```