---
title: 테이블
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-sdk-data-types-Table
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/table.py >}}




## <kbd>class</kbd> `Table`
Table 클래스는 표 형태의 데이터를 시각화하고 분석하는 데 사용됩니다.

일반적인 스프레드시트와 달리 Table은 다양한 유형의 데이터를 지원합니다: 스칼라 값, 문자열, numpy 배열, 그리고 대부분의 `wandb.data_types.Media`의 서브클래스들도 지원합니다. 즉, `Images`, `Video`, `Audio`와 같이 풍부하게 주석이 달린 미디어를 기존의 스칼라 값들과 함께 Table에 직접 삽입할 수 있습니다.

이 클래스는 W&B Tables https://docs.wandb.ai/guides/models/tables/ 를 생성하는 데 주요하게 사용됩니다.

### <kbd>method</kbd> `Table.__init__`

```python
__init__(
    columns=None,
    data=None,
    rows=None,
    dataframe=None,
    dtype=None,
    optional=True,
    allow_mixed_types=False,
    log_mode: Optional[Literal['IMMUTABLE', 'MUTABLE', 'INCREMENTAL']] = 'IMMUTABLE'
)
```

Table 오브젝트를 초기화합니다.

rows는 과거의 호환성을 위해 남아 있으나, 사용하지 않는 것이 권장됩니다. Table 클래스는 Pandas API처럼 동작하기 위해 data를 사용합니다.



**인수:**
 
 - `columns`:  (List[str]) 테이블의 각 열 이름. 기본값은 ["Input", "Output", "Expected"]입니다.
 - `data`:  (List[List[any]]) 2차원(행 단위) 값 배열입니다.
 - `dataframe`:  (pandas.DataFrame) 테이블 생성을 위해 사용하는 DataFrame 오브젝트입니다. 이 값이 지정되면 `data`와 `columns` 인수는 무시됩니다.
 - `rows`:  (List[List[any]]) 2차원(행 단위) 값 배열입니다.
 - `optional`:  (Union[bool,List[bool]]) `None` 값 허용 여부입니다. 기본값은 True입니다.
        - 하나의 bool 값이면, 생성시 지정한 모든 열에 대해 동일하게 적용됩니다.
        - bool 값들의 리스트라면, 각 열마다 별도로 적용되며, 리스트 길이는 `columns`의 길이와 같아야 합니다.
 - `allow_mixed_types`:  (bool) 열에 혼합 타입이 허용되는지 지정합니다(타입 유효성 검증 비활성화). 기본값은 False입니다.
 - `log_mode`:  Optional[str] Table에서 값이 변경될 때 로그를 남기는 방식을 지정합니다. 옵션:
        - "IMMUTABLE"(기본값): Table은 한 번만 로그가 가능하며, table이 변경된 이후의 추가 로그 시도는 무시됩니다.
        - "MUTABLE": Table이 변경된 후에도 재로그가 가능하며, 매번 새로운 artifact 버전이 생성됩니다.
        - "INCREMENTAL": Table의 데이터가 점진적으로 추가로 로그되며, 각 로그마다 새로 추가된 데이터만 포함하는 artifact entry가 생성됩니다.




---

### <kbd>method</kbd> `Table.add_column`

```python
add_column(name, data, optional=False)
```

새로운 데이터 컬럼을 테이블에 추가합니다.



**인수:**
 
 - `name`:  (str) - 컬럼의 고유한 이름
 - `data`:  (list | np.array) - 일관된 타입의 컬럼 데이터
 - `optional`:  (bool) - null 유사 값 허용 여부

---

### <kbd>method</kbd> `Table.add_computed_columns`

```python
add_computed_columns(fn)
```

기존 데이터를 기반으로 연산된 컬럼을 하나 이상 추가합니다.



**인수:**
 
 - `fn`:  ndx(int)와 row(dict) 두 개의 파라미터를 받을 수 있는 함수여야 하고, 해당 row의 새 컬럼 이름을 키로 하는 dict를 반환해야 합니다.
    - `ndx`: 행의 인덱스를 의미하는 정수입니다. `include_ndx`가 True로 설정된 경우에만 포함됩니다.
    - `row`: 기존 컬럼명을 키로 하는 사전입니다.

---

### <kbd>method</kbd> `Table.add_data`

```python
add_data(*data)
```

테이블에 새로운 데이터 행(row)을 추가합니다.

Table의 최대 행(row) 수는 `wandb.Table.MAX_ARTIFACT_ROWS`로 제한됩니다.

데이터의 길이는 테이블의 컬럼 개수와 일치해야 합니다.

---

### <kbd>method</kbd> `Table.add_row`

```python
add_row(*row)
```

더 이상 사용되지 않습니다. 대신 `Table.add_data` 메소드를 사용하세요.

---


### <kbd>method</kbd> `Table.cast`

```python
cast(col_name, dtype, optional=False)
```

특정 열을 지정한 데이터 타입으로 변환합니다.

일반적인 파이썬 클래스, 내부 W&B 타입, 예시 오브젝트(예: wandb.Image나 wandb.Classes 인스턴스) 등으로 지정 가능합니다.



**인수:**
 
 - `col_name` (str):  변환할 컬럼명
 - `dtype` (class, wandb.wandb_sdk.interface._dtypes.Type, any):  변환할 대상 dtype
 - `optional` (bool):  해당 열에 None 값 허용 여부

---


### <kbd>method</kbd> `Table.get_column`

```python
get_column(name, convert_to=None)
```

테이블의 특정 컬럼을 가져오며, 필요에 따라 NumPy 오브젝트로 변환할 수 있습니다.



**인수:**
 
 - `name`:  (str) - 컬럼명
 - `convert_to`:  (str, optional)
        - "numpy": 해당 데이터를 numpy 오브젝트로 변환합니다

---

### <kbd>method</kbd> `Table.get_dataframe`

```python
get_dataframe()
```

해당 테이블을 `pandas.DataFrame`형태로 반환합니다.

---

### <kbd>method</kbd> `Table.get_index`

```python
get_index()
```

다른 테이블에서 링크 생성을 위해 사용할 수 있는 행 인덱스 배열을 반환합니다.

---