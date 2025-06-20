---
title: Table
menu:
  reference:
    identifier: ko-ref-python-data-types-table
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L183-L909 >}}

표 형식의 데이터를 표시하고 분석하는 데 사용되는 Table 클래스입니다.

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

기존 스프레드시트와 달리 Tables는 다양한 유형의 데이터를 지원합니다. 스칼라 값, 문자열, numpy 배열 및 `wandb.data_types.Media`의 대부분의 서브클래스가 해당됩니다.
즉, 다른 기존 스칼라 값과 함께 `Images`, `Video`, `Audio` 및 기타 종류의 풍부하고 주석이 달린 미디어를 Tables에 직접 포함할 수 있습니다.

이 클래스는 UI에서 Table Visualizer를 생성하는 데 사용되는 기본 클래스입니다. https://docs.wandb.ai/guides/models/tables/ 를 참조하세요.

| 인수 |  |
| :--- | :--- |
|  `columns` |  (List[str]) 테이블의 열 이름입니다. 기본값은 ["Input", "Output", "Expected"]입니다. |
|  `data` |  (List[List[any]]) 값의 2D 행 중심 배열입니다. |
|  `dataframe` |  (pandas.DataFrame) 테이블을 만드는 데 사용되는 DataFrame 오브젝트입니다. 설정되면 `data` 및 `columns` 인수는 무시됩니다. |
|  `optional` |  (Union[bool,List[bool]]) `None` 값이 허용되는지 여부를 결정합니다. 기본값은 True입니다. 단일 bool 값인 경우 구성 시 지정된 모든 열에 대해 선택 사항이 적용됩니다. bool 값 목록인 경우 각 열에 선택 사항이 적용됩니다. `columns`와 길이가 같아야 합니다. 모든 열에 적용됩니다. bool 값 목록은 각 해당 열에 적용됩니다. |
|  `allow_mixed_types` |  (bool) 열에 혼합된 유형을 허용할지 여부를 결정합니다(유형 유효성 검사 비활성화). 기본값은 False입니다. |

## Methods

### `add_column`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L797-L836)

```python
add_column(
    name, data, optional=(False)
)
```

테이블에 데이터 열을 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) - 열의 고유한 이름 |
|  `data` |  (list | np.array) - 동종 데이터 열 |
|  `optional` |  (bool) - null과 유사한 값이 허용되는지 여부 |

### `add_computed_columns`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L887-L909)

```python
add_computed_columns(
    fn
)
```

기존 데이터를 기반으로 하나 이상의 계산된 열을 추가합니다.

| 인수 |  |
| :--- | :--- |
|  `fn` |  하나 또는 두 개의 파라미터, ndx (int) 및 row (dict)를 허용하는 함수입니다. 여기서 ndx는 행의 인덱스를 나타내는 정수입니다. `include_ndx`가 `True`로 설정된 경우에만 포함됩니다. `row`는 기존 열을 키로 사용하는 사전입니다. 이 함수는 해당 행에 대한 새 열을 나타내는 사전을 반환해야 합니다(새 열 이름을 키로 사용). |

### `add_data`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L423-L456)

```python
add_data(
    *data
)
```

테이블에 새 데이터 행을 추가합니다. 테이블의 최대 행 수는 `wandb.Table.MAX_ARTIFACT_ROWS`에 의해 결정됩니다.

데이터 길이는 테이블 열의 길이와 일치해야 합니다.

### `add_row`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L418-L421)

```python
add_row(
    *row
)
```

더 이상 사용되지 않습니다. 대신 add_data를 사용하세요.

### `cast`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L315-L371)

```python
cast(
    col_name, dtype, optional=(False)
)
```

열을 특정 데이터 유형으로 캐스팅합니다.

이는 일반 Python 클래스, 내부 W&B 유형 또는
wandb.Image 또는 wandb.Classes의 인스턴스와 같은 예제 오브젝트 중 하나일 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `col_name` |  (str) - 캐스팅할 열의 이름입니다. |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - 대상 dtype입니다. |
|  `optional` |  (bool) - 열에서 None을 허용해야 하는지 여부입니다. |

### `get_column`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L838-L861)

```python
get_column(
    name, convert_to=None
)
```

테이블에서 열을 검색하고 선택적으로 NumPy 오브젝트로 변환합니다.

| 인수 |  |
| :--- | :--- |
|  `name` |  (str) - 열의 이름 |
|  `convert_to` |  (str, optional) - "numpy": 기본 데이터를 numpy 오브젝트로 변환합니다. |

### `get_dataframe`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L872-L878)

```python
get_dataframe()
```

테이블의 `pandas.DataFrame`을 반환합니다.

### `get_index`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L863-L870)

```python
get_index()
```

다른 테이블에서 링크를 만드는 데 사용할 행 인덱스 배열을 반환합니다.

### `index_ref`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L880-L885)

```python
index_ref(
    index
)
```

테이블에서 행의 인덱스에 대한 참조를 가져옵니다.

### `iterrows`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L674-L688)

```python
iterrows()
```

행의 인덱스 및 관련 데이터를 표시하여 테이블 데이터를 행별로 반환합니다.

| Yields |  |
| :--- | :--- |

***

index : int
행의 인덱스입니다. 다른 W&B 테이블에서 이 값을 사용하면
테이블 간에 관계가 자동으로 구축됩니다.
row : List[any]
행의 데이터입니다.

### `set_fk`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L695-L699)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L690-L693)

```python
set_pk(
    col_name
)
```

| Class Variables |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |
