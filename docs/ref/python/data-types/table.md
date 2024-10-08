# Table

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L151-L877' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Table 클래스는 표형 데이터의 표시와 분석에 사용됩니다.

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

전통적인 스프레드시트와 달리, Tables는 다양한 유형의 데이터를 지원합니다:
스칼라 값, 문자열, numpy 배열, 그리고 대부분의 `wandb.data_types.Media`의 서브클래스.
이는 `Images`, `Video`, `Audio` 등의 풍부하고 주석이 달린 미디어를
기본적인 스칼라 값과 함께 Tables에 직접 삽입할 수 있음을 의미합니다.

이 클래스는 UI에서 Table Visualizer를 생성하는데 주로 사용됩니다: /guides/data-vis/tables.

| 인수 |  |
| :--- | :--- |
|  `columns` |  (List[str]) 테이블의 열 이름. 기본값은 ["Input", "Output", "Expected"]입니다. |
|  `data` |  (List[List[any]]) 2D 행지향 배열입니다. |
|  `dataframe` |  (pandas.DataFrame) 테이블을 생성하는 데 사용되는 DataFrame 오브젝트. 설정 시, `data` 및 `columns` 인수는 무시됩니다. |
|  `optional` |  (Union[bool,List[bool]]) `None` 값이 허용될 지 여부를 결정합니다. 기본값은 True입니다 - 단일 bool 값인 경우, 생성 시 지정된 모든 열에 대해 선택적 속성이 적용됨 - bool 값 목록의 경우, 선택적 속성이 각 열에 적용됨 - `columns`와 동일한 길이여야 하며 모든 열에 적용됨. |
|  `allow_mixed_types` |  (bool) 열에 다양한 유형이 허용될 지 여부를 결정합니다 (유형 검증 비활성화). 기본값은 False입니다. |

## 메소드

### `add_column`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L765-L804)

```python
add_column(
    name, data, optional=(False)
)
```

테이블에 데이터 열을 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) - 열의 고유 이름 |
|  `data` |  (list | np.array) - 균일한 데이터의 열 |
|  `optional` |  (bool) - null-유사 값을 허용할 지 여부 |

### `add_computed_columns`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L855-L877)

```python
add_computed_columns(
    fn
)
```

기존 데이터 기반으로 계산된 열을 하나 이상 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `fn` |  ndx (int) 및 row (dict)의 두 인수를 받는 함수로, 해당 행을 위한 새 열을 나타내는 사전을 반환하는 것으로 기대됩니다. 반환되는 사전은 새 열 이름으로 키가 지정됩니다. `ndx`는 행의 인덱스를 나타내는 정수입니다. `include_ndx`가 `True`로 설정된 경우에만 포함됩니다. `row`는 기존 열로 키가 지정된 사전입니다. |

### `add_data`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L391-L424)

```python
add_data(
    *data
)
```

테이블에 새로운 행의 데이터를 추가합니다. 테이블의 최대 행 수는 `wandb.Table.MAX_ARTIFACT_ROWS`에 의해 결정됩니다.

데이터 길이는 테이블 열의 길이와 일치해야 합니다.

### `add_row`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L386-L389)

```python
add_row(
    *row
)
```

사용 중단됨; 대신 add_data를 사용합니다.

### `cast`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L283-L339)

```python
cast(
    col_name, dtype, optional=(False)
)
```

열을 특정 데이터 유형으로 변환합니다.

이것은 일반 Python 클래스, 내부 W&B 유형 또는 wandb.Image나 wandb.Classes의 인스턴스와 같은 예제 오브젝트일 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `col_name` |  (str) - 변환할 열의 이름 |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - 목표 데이터 유형 |
|  `optional` |  (bool) - 열에 `None`을 허용할 지 여부 |

### `get_column`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L806-L829)

```python
get_column(
    name, convert_to=None
)
```

테이블에서 열을 가져오고 선택적으로 NumPy 오브젝트로 변환합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) - 열의 이름 |
|  `convert_to` |  (str, optional) - "numpy": 기본 데이터를 numpy 오브젝트로 변환합니다 |

### `get_dataframe`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L840-L846)

```python
get_dataframe()
```

테이블의 `pandas.DataFrame`을 반환합니다.

### `get_index`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L831-L838)

```python
get_index()
```

다른 테이블에서 링크를 생성하기 위한 행 인덱스 배열을 반환합니다.

### `index_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L848-L853)

```python
index_ref(
    index
)
```

테이블의 행의 인덱스에 대한 참조를 가져옵니다.

### `iterrows`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L642-L656)

```python
iterrows()
```

행 별로 테이블 데이터를 반환하며, 행의 인덱스와 관련된 데이터를 보여줍니다.

| 반환값 |  |
| :--- | :--- |

***

index : int
행의 인덱스입니다. 이 값을 다른 W&B 테이블에서 사용하면
자동으로 테이블 간의 관계가 형성됩니다.
row : List[any]
행의 데이터입니다.

### `set_fk`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L663-L667)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/data_types.py#L658-L661)

```python
set_pk(
    col_name
)
```

| 클래스 변수 |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |