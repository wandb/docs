---
title: テーブル
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-data-types-Table
object_type: python_sdk_data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/table.py >}}




## <kbd>class</kbd> `Table`
Table クラスは、表形式データの表示や解析に使用されます。

従来のスプレッドシートとは異なり、Tables ではスカラー値や文字列、numpy配列、さらに `wandb.data_types.Media` のほとんどのサブクラスなど、さまざまな型のデータをサポートしています。これにより、`Images`、`Video`、`Audio` といったリッチで注釈付きのメディアも、他の従来型スカラー値とともに、直接 Tables 内に埋め込むことができます。

このクラスは W&B Tables https://docs.wandb.ai/guides/models/tables/ を生成する際に主に使われます。

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

`rows` はレガシー対応のために残されていますが、使用は推奨されません。Table クラスは Pandas API を模倣するために data を利用します。

**引数:**
 
 - `columns`:  (List[str]) テーブルのカラム名。デフォルトは ["Input", "Output", "Expected"] です。
 - `data`:  (List[List[any]]) 2次元の行指向配列データ。
 - `dataframe`:  (pandas.DataFrame) テーブル作成に使用される DataFrame オブジェクト。これが指定された場合、`data` と `columns` 引数は無視されます。
 - `rows`:  (List[List[any]]) 2次元の行指向配列データ。
 - `optional`:  (Union[bool, List[bool]]) `None` 値を許可するかどうか。デフォルトは True。
        - 単一の bool 値の場合、コンストラクタで指定されたすべてのカラムに適用されます。
        - bool 値のリストの場合、それぞれのカラムに個別に適用されます。リストの長さは `columns` の数と一致させる必要があります。
 - `allow_mixed_types`:  (bool) カラムで異なる型を許可するか（型検証を無効化）。デフォルトは False。
 - `log_mode`:  Optional[str] Table が変更された際のログ方法を制御します。オプション:
        - "IMMUTABLE"（デフォルト）: Table は一度しかログできず、変更後の再ログは無視されます。
        - "MUTABLE": 変更後も再度ログ可能で、その都度新しい artifact バージョンが作成されます。
        - "INCREMENTAL": Table データが段階的に記録され、各ログ時に前回以降の新データのみが新しい artifact エントリとして追加されます。

---

### <kbd>method</kbd> `Table.add_column`

```python
add_column(name, data, optional=False)
```

新しいカラムデータをテーブルに追加します。

**引数:**
 
 - `name`:  (str) - カラムの一意な名前
 - `data`:  (list | np.array) - 同じ型のみからなるカラムデータ
 - `optional`:  (bool) - null や None 相当の値が許可される場合

---

### <kbd>method</kbd> `Table.add_computed_columns`

```python
add_computed_columns(fn)
```

既存データをもとに計算されたカラムを 1つ以上追加します。

**引数:**
 
 - `fn`:  1 または 2 つの引数（ndx (int), row (dict)）を受け取る関数。各行に対して新しいカラム名をキーにした辞書を返す必要があります。
    - `ndx` は行番号（index）の整数。`include_ndx` が True の場合に渡されます。
    - `row` は既存カラム名をキーに持つ辞書。

---

### <kbd>method</kbd> `Table.add_data`

```python
add_data(*data)
```

新しい行データをテーブルへ追加します。

テーブルに保持できる最大行数は `wandb.Table.MAX_ARTIFACT_ROWS` で決まっています。

追加データの長さは、テーブルのカラム数と同じである必要があります。

---

### <kbd>method</kbd> `Table.add_row`

```python
add_row(*row)
```

非推奨です。代わりに `Table.add_data` メソッドをご利用ください。

---

### <kbd>method</kbd> `Table.cast`

```python
cast(col_name, dtype, optional=False)
```

指定したカラムのデータ型を変更します。

これは通常の Python クラスや、内部の W&B 型、または例として wandb.Image や wandb.Classes のインスタンスなどを指定可能です。

**引数:**
 
 - `col_name` (str):  型変換するカラム名
 - `dtype` (class, wandb.wandb_sdk.interface._dtypes.Type, any):  変更先の dtype
 - `optional` (bool):  カラムに None を許可するかどうか

---

### <kbd>method</kbd> `Table.get_column`

```python
get_column(name, convert_to=None)
```

指定したカラムを取得し、必要に応じて NumPy オブジェクトへ変換します。

**引数:**
 
 - `name`:  (str) - カラム名
 - `convert_to`:  (str, optional)
        - "numpy": データを numpy オブジェクトに変換します

---

### <kbd>method</kbd> `Table.get_dataframe`

```python
get_dataframe()
```

テーブル全体を `pandas.DataFrame` として返します。

---

### <kbd>method</kbd> `Table.get_index`

```python
get_index()
```

他のテーブルでリンク作成に利用できる行インデックスの配列を返します。

---