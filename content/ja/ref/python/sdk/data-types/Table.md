---
title: 表
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Table
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/table.py >}}




## <kbd>class</kbd> `Table`
テーブル形式のデータを表示・分析するための Table クラスです。 

従来のスプレッドシートと異なり、Table は多様な型のデータをサポートします。たとえばスカラー値、文字列、NumPy 配列、そして `wandb.data_types.Media` のほとんどのサブクラスです。つまり、`Images`、`Video`、`Audio` などの豊富で注釈付きのメディアを、従来のスカラー値と並べて Table に直接埋め込めます。 

このクラスは W&B Tables を生成するための主要なクラスです https://docs.wandb.ai/guides/models/tables/. 

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

Table オブジェクトを初期化します。 

rows は後方互換のために残されていますが、使用しないでください。Table クラスは Pandas API に合わせて data を使用します。 



**Args:**
 
 - `columns`:  (List[str]) テーブル内の列名。デフォルトは ["Input", "Output", "Expected"]。 
 - `data`:  (List[List[any]]) 値の 2 次元の行指向配列。 
 - `dataframe`:  (pandas.DataFrame) テーブルの作成に用いる DataFrame オブジェクト。設定した場合は `data` と `columns` の引数は無視されます。 
 - `rows`:  (List[List[any]]) 値の 2 次元の行指向配列。 
 - `optional`:  (Union[bool,List[bool]]) `None` の値を許可するかどうか。デフォルトは True 
        - 単一の bool の場合、構築時に指定されたすべての列に対して同じ可否を適用します 
        - bool のリストの場合、各列に対して個別に可否を適用します（`columns` と同じ長さである必要があります） 
 - `allow_mixed_types`:  (bool) 列に混在した型を許可するかどうか（型検証を無効化）。デフォルトは False 
 - `log_mode`:  Optional[str] 変更があったときに Table をどのようにログするかを制御します。オプション: 
        - "IMMUTABLE"（デフォルト）: Table は 1 度だけログ可能。以降、テーブルが変更された後のログは何もしません。 
        - "MUTABLE": 変更後に再ログ可能で、そのたびに新しい artifact の version を作成します。 
        - "INCREMENTAL": Table のデータを増分でログし、各ログで前回以降の新しいデータを含む新しい artifact のエントリを作成します。 




---

### <kbd>method</kbd> `Table.add_column`

```python
add_column(name, data, optional=False)
```

テーブルに 1 列のデータを追加します。 



**Args:**
 
 - `name`:  (str) 列の一意な名前 
 - `data`:  (list | np.array) 同質なデータの 1 列 
 - `optional`:  (bool) null のような値を許可するかどうか 

---

### <kbd>method</kbd> `Table.add_computed_columns`

```python
add_computed_columns(fn)
```

既存のデータに基づいて 1 列以上の計算列を追加します。 



**Args:**
 
 - `fn`:  ndx (int) と row (dict) の 1 つまたは 2 つの引数を受け取り、その行に対して新しい列を表す dict（新しい列名をキーにする）を返す関数。 
    - `ndx` は行のインデックスを表す整数。`include_ndx` が `True` に設定されている場合にのみ含まれます。 
    - `row` は既存の列をキーにした 辞書。 

---

### <kbd>method</kbd> `Table.add_data`

```python
add_data(*data)
```

テーブルに新しい 1 行のデータを追加します。 

テーブルの最大行数は `wandb.Table.MAX_ARTIFACT_ROWS` によって決まります。 

渡すデータの長さは、テーブルの列数と一致している必要があります。 

---

### <kbd>method</kbd> `Table.add_row`

```python
add_row(*row)
```

非推奨。代わりに `Table.add_data` メソッドを使用してください。 

---


### <kbd>method</kbd> `Table.cast`

```python
cast(col_name, dtype, optional=False)
```

列を特定のデータ型にキャストします。 

通常の Python クラス、W&B の内部型、または wandb.Image や wandb.Classes のインスタンスのような例となるオブジェクトのいずれかを指定できます。 



**Args:**
 
 - `col_name` (str):  キャストする列の名前。 
 - `dtype` (class, wandb.wandb_sdk.interface._dtypes.Type, any):  目的の dtype。 
 - `optional` (bool):  その列で None を許可するかどうか。 

---


### <kbd>method</kbd> `Table.get_column`

```python
get_column(name, convert_to=None)
```

テーブルから列を取得し、必要に応じて NumPy オブジェクトに変換します。 



**Args:**
 
 - `name`:  (str) 列の名前 
 - `convert_to`:  (str, optional) 
        - "numpy": 基になるデータを numpy オブジェクトに変換します 

---

### <kbd>method</kbd> `Table.get_dataframe`

```python
get_dataframe()
```

テーブルを `pandas.DataFrame` として返します。 

---

### <kbd>method</kbd> `Table.get_index`

```python
get_index()
```

他のテーブルでリンクを作成するために使用できる、行インデックスの配列を返します。 

---