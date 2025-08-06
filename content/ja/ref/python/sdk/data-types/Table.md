---
title: テーブル
object_type: python_sdk_data_type
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/table.py >}}



## <kbd>class</kbd> `Table`
Table クラスは、表形式のデータを表示・分析するためのクラスです。

従来のスプレッドシートとは異なり、Tables ではスカラー値や文字列、numpy 配列、さらに `wandb.data_types.Media` のほとんどのサブクラスなど、さまざまな種類のデータを扱うことができます。これにより、`Images` や `Video`、`Audio` など様々なリッチでアノテートされたメディアを、通常のスカラー値と並べて Tables 内に組み込むことが可能になります。

このクラスは、W&B Tables https://docs.wandb.ai/guides/models/tables/ を作成する際の主要なクラスです。

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

rows 引数は後方互換のために利用可能ですが、基本的には使用しないでください。Table クラスは data により Pandas API を模倣しています。

**引数:**
 
 - `columns`:  (List[str]) テーブルのカラム名。省略時は ["Input", "Output", "Expected"] です。
 - `data`:  (List[List[any]]) 2次元の行単位で格納された値の配列。
 - `dataframe`:  (pandas.DataFrame) テーブル作成に利用される DataFrame オブジェクト。これが指定されている場合は、`data` および `columns` の引数は無視されます。
 - `rows`:  (List[List[any]]) 2次元の行単位で格納された値の配列。
 - `optional`:  (Union[bool, List[bool]]) `None` 値を許可するかどうかを決定します。デフォルトは True。
        - 単一の bool 値の場合、構築時に指定されたすべてのカラムに対して同じ optional 設定が適用されます。
        - bool 値のリストの場合、それぞれのカラムごとに optional 設定が適用されます（`columns` と同じ長さのリストである必要があります）。
 - `allow_mixed_types`: (bool) カラム内で異なる型を許可するかどうか（型検証を無効化）。デフォルトは False です。
 - `log_mode`: Optional[str] テーブルが変更された際にどのようにログされるかを制御します。選択肢:
        - "IMMUTABLE"（デフォルト）: テーブルは一度だけログ可能で、その後の変更後の再ログは行われません。
        - "MUTABLE": 変更後も再ログ可能で、その都度新しい artifact バージョンが作成されます。
        - "INCREMENTAL": テーブルデータを段階的にログし、前回のログ以降の新しいデータのみを含む artifact エントリーが各ログで作成されます。

---

### <kbd>method</kbd> `Table.add_column`

```python
add_column(name, data, optional=False)
```

テーブルに新しいカラム（列）データを追加します。

**引数:**
 
 - `name`:  (str) - カラムのユニークな名前
 - `data`:  (list | np.array) - 同一型データのカラム
 - `optional`:  (bool) - null のような値を許容するかどうか

---

### <kbd>method</kbd> `Table.add_computed_columns`

```python
add_computed_columns(fn)
```

既存データをもとに、一つまたは複数の計算済みカラムを追加します。

**引数:**
 
 - `fn`: ndx（int）と row（dict）の1つまたは2つの引数を受け取る関数で、各行に対して新しいカラム名をキーに持つ辞書を返すことが期待されます。
    - `ndx` は行のインデックスを表す整数です。`include_ndx` が True の場合のみ含まれます。
    - `row` は既存のカラム名をキーに持つ辞書です。

---

### <kbd>method</kbd> `Table.add_data`

```python
add_data(*data)
```

新しい行データをテーブルに追加します。

テーブルに登録できる最大行数は `wandb.Table.MAX_ARTIFACT_ROWS` により制限されます。

追加するデータの要素数はテーブルのカラム数と一致する必要があります。

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

指定したカラムを特定のデータ型に変換します。

通常の Python クラス、W&B 独自型、または wandb.Image や wandb.Classes のような例示的オブジェクトとして指定することも可能です。

**引数:**
 
 - `col_name` (str):  変換したいカラム名
 - `dtype` (class, wandb.wandb_sdk.interface._dtypes.Type, any):  変換したい型
 - `optional` (bool):  Nones を許容するかどうか

---

### <kbd>method</kbd> `Table.get_column`

```python
get_column(name, convert_to=None)
```

テーブルから指定したカラムを取得し、必要に応じて NumPy オブジェクトへ変換します。

**引数:**
 
 - `name`:  (str) - カラム名
 - `convert_to`:  (str, オプション) 
        - "numpy" を指定するとデータを numpy オブジェクトに変換

---

### <kbd>method</kbd> `Table.get_dataframe`

```python
get_dataframe()
```

テーブルを `pandas.DataFrame` 形式で返します。

---

### <kbd>method</kbd> `Table.get_index`

```python
get_index()
```

他のテーブルとリンクを作成するために使用できる行インデックスの配列を返します。

---