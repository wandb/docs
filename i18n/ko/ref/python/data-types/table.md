
# 테이블

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L150-L873' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


테이블 데이터를 표시하고 분석하는 데 사용되는 Table 클래스입니다.

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

전통적인 스프레드시트와 달리, 테이블은 다양한 유형의 데이터를 지원합니다: 스칼라 값, 문자열, numpy 배열 및 `wandb.data_types.Media`의 대부분의 서브클래스. 이는 Images, Video, Audio 및 기타 풍부한 주석이 달린 미디어를 다른 전통적인 스칼라 값과 함께 테이블에 직접 포함할 수 있음을 의미합니다.

이 클래스는 UI에서 Table Visualizer를 생성하는 데 사용되는 주요 클래스입니다: https://docs.wandb.ai/guides/data-vis/tables.

| 인수 |  |
| :--- | :--- |
|  `columns` |  (List[str]) 테이블의 열 이름. 기본값은 ["Input", "Output", "Expected"]. |
|  `data` |  (List[List[any]]) 값의 2D 행 기반 배열. |
|  `dataframe` |  (pandas.DataFrame) 테이블을 생성하는 데 사용된 DataFrame 객체. 설정될 때, `data` 및 `columns` 인수는 무시됩니다. |
|  `optional` |  (Union[bool,List[bool]]) `None` 값이 허용되는지 결정합니다. 기본값은 True - 단일 bool 값이면, 모든 열에 대해 선택 사항이 구축 시간에 지정된 열에 대해 적용됩니다 - bool 값의 목록이면, 각 열에 대해 선택 사항이 적용됩니다 - `columns`에 적용되는 길이와 동일해야 합니다. bool 값의 목록은 각각의 열에 적용됩니다. |
|  `allow_mixed_types` |  (bool) 열이 혼합 유형을 가질 수 있는지 여부를 결정합니다(유형 검증 비활성화). 기본값은 False |

## 메소드

### `add_column`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L761-L800)

```python
add_column(
    name, data, optional=(False)
)
```

테이블에 데이터 열을 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) - 열의 고유 이름 |
|  `data` |  (list | np.array) - 동질의 데이터 열 |
|  `optional` |  (bool) - null과 같은 값이 허용되는지 여부 |

### `add_computed_columns`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L851-L873)

```python
add_computed_columns(
    fn
)
```

기존 데이터를 기반으로 하나 이상의 계산된 열을 추가합니다.

| Args |  |
| :--- | :--- |
|  `fn` |  하나 또는 두 개의 파라미터(ndx(int) 및 row(dict))를 받아들이는 함수로, 해당 행에 대한 새 열을 나타내는 dict를 반환해야 합니다. 새 열 이름으로 키가 지정됩니다. `ndx`는 행의 인덱스를 나타내는 정수입니다. `include_ndx`가 `True`로 설정된 경우에만 포함됩니다. `row`는 기존 열로 키가 지정된 딕셔너리입니다 |

### `add_data`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L387-L420)

```python
add_data(
    *data
)
```

테이블에 새 데이터 행을 추가합니다. 테이블의 최대 행 수는 `wandb.Table.MAX_ARTIFACT_ROWS`에 의해 결정됩니다.

데이터의 길이는 테이블 열의 길이와 일치해야 합니다.

### `add_row`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L382-L385)

```python
add_row(
    *row
)
```

사용되지 않음; 대신 add_data를 사용하세요.

### `cast`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L282-L335)

```python
cast(
    col_name, dtype, optional=(False)
)
```

열을 특정 데이터 유형으로 캐스트합니다. 이는 일반 파이썬 클래스, 내부 W&B 유형 또는 wandb.Image 또는 wandb.Classes의 인스턴스와 같은 예제 오브젝트 중 하나일 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `col_name` |  (str) - 캐스트할 열의 이름. |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - 목표 dtype. |
|  `optional` |  (bool) - 열이 None을 허용해야 하는지 여부. |

### `get_column`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L802-L825)

```python
get_column(
    name, convert_to=None
)
```

테이블에서 열을 검색하고 선택적으로 NumPy 객체로 변환합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) - 열의 이름 |
|  `convert_to` |  (str, optional) - "numpy": 기본 데이터를 numpy 객체로 변환합니다 |

### `get_dataframe`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L836-L842)

```python
get_dataframe()
```

테이블의 `pandas.DataFrame`을 반환합니다.

### `get_index`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L827-L834)

```python
get_index()
```

다른 테이블에서 링크를 생성하기 위해 사용되는 행 인덱스의 배열을 반환합니다.

### `index_ref`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L844-L849)

```python
index_ref(
    index
)
```

테이블에서 행의 인덱스 참조를 가져옵니다.

### `iterrows`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L638-L652)

```python
iterrows()
```

행의 인덱스와 관련 데이터를 보여주며 테이블 데이터를 행별로 반환합니다.

| Yields |  |
| :--- | :--- |

***

index : int
행의 인덱스. 이 값을 다른 W&B 테이블에서 사용하면
테이블 간의 관계가 자동으로 구축됩니다
row : List[any]
행의 데이터.

### `set_fk`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L659-L663)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/data_types.py#L654-L657)

```python
set_pk(
    col_name
)
```

| 클래스 변수 |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |