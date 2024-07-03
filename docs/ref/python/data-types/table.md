# Table

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L150-L876' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Tableクラスは表形式のデータを表示・分析するために使用されます。

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

従来のスプレッドシートとは異なり、Tablesは以下のようなさまざまなタイプのデータをサポートしています：
スカラー値、文字列、numpy配列、および`wandb.data_types.Media`のほとんどのサブクラス。このため、`Images`、`Video`、`Audio`などのリッチで注釈付きのメディアを、他の従来のスカラー値と同様に直接Tablesに埋め込むことができます。

このクラスは、UIでTable Visualizerを生成するための主要なクラスです: https://docs.wandb.ai/guides/data-vis/tables.

| 引数 |  |
| :--- | :--- |
|  `columns` |  (List[str]) テーブルのカラム名。デフォルトは ["Input", "Output", "Expected"]。 |
|  `data` |  (List[List[any]]) 行指向の2次元配列。 |
|  `dataframe` |  (pandas.DataFrame) Tableを作成するために使用されるDataFrameオブジェクト。設定されている場合、`data`および`columns`引数は無視されます。|
|  `optional` |  (Union[bool,List[bool]]) `None` 値を許可するかどうかを決定します。デフォルトはTrue - 単一のbool値の場合、構築時に指定されたすべての列に対してオプション性が適用されます - bool値のリストの場合、それぞれの列に対してオプション性が適用されます。 |
|  `allow_mixed_types` |  (bool) 列に混合タイプが許可されるかどうかを決定します（タイプ検証を無効にします）。デフォルトはFalse。 |

## メソッド

### `add_column`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L764-L803)

```python
add_column(
    name, data, optional=(False)
)
```

Tableに新しいカラムデータを追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) - カラムの一意の名前 |
|  `data` |  (list | np.array) - 同質データのカラム |
|  `optional` |  (bool) - nullのような値が許可されているかどうか |

### `add_computed_columns`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L854-L876)

```python
add_computed_columns(
    fn
)
```

既存のデータに基づいて1つまたは複数の計算カラムを追加します。

| 引数 |  |
| :--- | :--- |
|  `fn` |  一つまたは二つのパラメータ（ndx（int）およびrow（dict））を受け取り、その行の新しいカラムを表すdictを返す関数。新しいカラム名でキー付けされます。`ndx`は行のインデックスを表す整数です。`include_ndx`が`True`に設定されている場合にのみ含まれます。`row`は既存のカラムでキー付けされた辞書です。|

### `add_data`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L390-L423)

```python
add_data(
    *data
)
```

Tableに新しい行データを追加します。テーブルの最大行数は`wandb.Table.MAX_ARTIFACT_ROWS`で決定されます。

データの長さはテーブルカラムの長さと一致する必要があります。

### `add_row`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L385-L388)

```python
add_row(
    *row
)
```

非推奨; 代わりにadd_dataを使用してください。

### `cast`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L282-L338)

```python
cast(
    col_name, dtype, optional=(False)
)
```

カラムを特定のデータ型にキャストします。

これには通常のpythonクラス、内部W&B型、またはwandb.Imageやwandb.Classesのインスタンスのようなオブジェクトが含まれます。

| 引数 |  |
| :--- | :--- |
|  `col_name` |  (str) - キャストするカラムの名前。 |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - 目標とするデータ型。 |
|  `optional` |  (bool) - カラムがNoneを許可するかどうか。 |

### `get_column`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L805-L828)

```python
get_column(
    name, convert_to=None
)
```

テーブルからカラムを取得し、任意でNumPyオブジェクトに変換します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) - カラムの名前 |
|  `convert_to` |  (str, optional) - "numpy": データをNumPyオブジェクトに変換します |

### `get_dataframe`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L839-L845)

```python
get_dataframe()
```

テーブルの`pandas.DataFrame`を返します。

### `get_index`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L830-L837)

```python
get_index()
```

他のテーブルでリンクを作成するために使用される行インデックスの配列を返します。

### `index_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L847-L852)

```python
index_ref(
    index
)
```

Tableの行のインデックスの参照を取得します。

### `iterrows`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L641-L655)

```python
iterrows()
```

行データとそのインデックスを返します。

| 戻り値 |  |
| :--- | :--- |

***

index : int
行のインデックス。この値を他のW&Bテーブルで使用することで、テーブル間の関係が自動的に構築されます。
row : List[any]
行データ。

### `set_fk`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L662-L666)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/data_types.py#L657-L660)

```python
set_pk(
    col_name
)
```

| クラス変数 |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |