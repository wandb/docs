
# Table

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L150-L876' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Table クラスは、表形式のデータを表示および分析するために使用されます。

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

従来のスプレッドシートとは異なり、Tables は多数のデータタイプをサポートします:
スカラー値、文字列、numpy 配列、そして `wandb.data_types.Media` の多くのサブクラス。
これにより、`Images`、`Video`、`Audio`、およびその他のリッチで注釈付きメディアを
従来のスカラー値と並べて直接 Table に埋め込むことができます。

このクラスは、UI で Table Visualizer を生成するための主要なクラスです: https://docs.wandb.ai/guides/data-vis/tables.

| 引数 |  |
| :--- | :--- |
|  `columns` |  (List[str]) テーブルの列名。デフォルトは ["Input", "Output", "Expected"]。 |
|  `data` |  (List[List[any]]) 値の2次元行指向配列。 |
|  `dataframe` |  (pandas.DataFrame) テーブルを作成するために使用される DataFrame オブジェクト。設定されている場合、`data` および `columns` 引数は無視されます。 |
|  `optional` |  (Union[bool,List[bool]]) `None` 値が許可されるかどうかを決定します。デフォルトは True - 単一の bool 値の場合、構築時に指定されたすべての列に optionality が適用されます - bool 値のリストの場合、各列に optionality が適用されます - `columns` と同じ長さである必要があります。bool 値のリストは、それぞれの列に適用されます。 |
|  `allow_mixed_types` |  (bool) 列に混合タイプが許可されるかどうかを決定します（タイプ検証を無効にします）。デフォルトは False |

## メソッド

### `add_column`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L764-L803)

```python
add_column(
    name, data, optional=(False)
)
```

データの列をテーブルに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) - 列の一意の名前 |
|  `data` |  (list | np.array) - 同種データの列 |
|  `optional` |  (bool) - null のような値が許可されるかどうか |

### `add_computed_columns`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L854-L876)

```python
add_computed_columns(
    fn
)
```

既存のデータに基づいて1つ以上の計算列を追加します。

| 引数 |  |
| :--- | :--- |
|  `fn` |  1つまたは2つのパラメータ、ndx (int) および row (辞書) を受け取り、その行の新しい列を新しい列名でキー指定した辞書を返す関数。`ndx` はその行のインデックスを表す整数。`include_ndx` が `True` に設定されている場合のみ含まれます。`row` は既存の列でキー指定された辞書です |

### `add_data`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L390-L423)

```python
add_data(
    *data
)
```

新しいデータ行をテーブルに追加します。テーブル内の行の最大数は `wandb.Table.MAX_ARTIFACT_ROWS` によって決定されます。

データの長さはテーブルの列の長さと一致する必要があります。

### `add_row`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L385-L388)

```python
add_row(
    *row
)
```

廃止予定; 代わりに add_data を使用してください。

### `cast`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L282-L338)

```python
cast(
    col_name, dtype, optional=(False)
)
```

列を特定のデータタイプにキャストします。

これには通常の Python クラス、内部の W&B タイプ、または wandb.Image や wandb.Classes のインスタンス例などのオブジェクトが含まれます。

| 引数 |  |
| :--- | :--- |
|  `col_name` |  (str) - キャストする列の名前。 |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - 目標の dtype。 |
|  `optional` |  (bool) - 列が None を許可するかどうか。 |

### `get_column`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L805-L828)

```python
get_column(
    name, convert_to=None
)
```

テーブルから列を取得し、オプションで NumPy オブジェクトに変換します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) - 列の名前 |
|  `convert_to` |  (str, optional) - "numpy": 基となるデータを numpy オブジェクトに変換します |

### `get_dataframe`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L839-L845)

```python
get_dataframe()
```

テーブルの `pandas.DataFrame` を返します。

### `get_index`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L830-L837)

```python
get_index()
```

他のテーブルでリンクを作成するために使用される行インデックスの配列を返します。

### `index_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L847-L852)

```python
index_ref(
    index
)
```

テーブル内の行のインデックスの参照を取得します。

### `iterrows`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L641-L655)

```python
iterrows()
```

行ごとにテーブルデータを返し、行のインデックスと関連するデータを表示します。

| 戻り値 |  |
| :--- | :--- |

***

index : int
行のインデックス。他の W&B テーブルでこの値を使用すると、テーブル間の関係が自動的に構築されます
row : List[any]
行のデータ。

### `set_fk`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L662-L666)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/data_types.py#L657-L660)

```python
set_pk(
    col_name
)
```

| クラス変数 |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |