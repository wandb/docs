---
title: テーブル
menu:
  reference:
    identifier: ja-ref-python-data-types-table
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L183-L909 >}}

Table クラスは表形式のデータを表示および分析するために使用されます。

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

従来のスプレッドシートとは異なり、Tables は多くの種類のデータをサポートしています:
スカラー値、文字列、numpy 配列、および `wandb.data_types.Media` のほとんどのサブクラス。
これにより、`Images`、`Video`、`Audio`、および他の種類のリッチで注釈のあるメディアを
従来のスカラー値と並べて Tables に直接埋め込むことができます。

このクラスは、UI の Table Visualizer を生成するために使用される主要なクラスです: https://docs.wandb.ai/guides/data-vis/tables.

| 引数 |  |
| :--- | :--- |
|  `columns` |  (List[str]) テーブル内の列の名前。デフォルトは["Input", "Output", "Expected"]です。 |
|  `data` |  (List[List[any]]) 2D 行指向の配列。 |
|  `dataframe` |  (pandas.DataFrame) テーブルの作成に使用される DataFrame オブジェクト。設定されている場合、`data` と `columns` 引数は無視されます。 |
|  `optional` |  (Union[bool,List[bool]]) `None` の値を許可するかどうかを決定します。デフォルトは True です - 単一の bool 値が指定された場合、構築時に指定されたすべての列において任意性が確保されます - bool 値のリストである場合、各列に適用される任意性が適用されます - `columns` と同じ長さでなければなりません。bool 値のリストはそれぞれの列に適用されます。 |
|  `allow_mixed_types` |  (bool) 列に混合タイプを許可するかどうかを決定します（タイプ検証を無効にします）。デフォルトは False です。 |

## メソッド

### `add_column`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L797-L836)

```python
add_column(
    name, data, optional=(False)
)
```

テーブルにデータ列を追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) - 列の一意の名前 |
|  `data` |  (list | np.array) - 同種のデータの列 |
|  `optional` |  (bool) - null 値が許可されるかどうか |

### `add_computed_columns`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L887-L909)

```python
add_computed_columns(
    fn
)
```

既存のデータに基づいて1つ以上の計算列を追加します。

| 引数 |  |
| :--- | :--- |
|  `fn` |  ndx（int）および row（dict）という1つまたは2つのパラメータを受け取り、新しい列のキーを新しい列名として指定した辞書を返す関数です。`ndx` は行のインデックスを示す整数です。`include_ndx` が `True` に設定されている場合にのみ含まれます。`row` は既存の列にキー付けされた辞書です。 |

### `add_data`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L423-L456)

```python
add_data(
    *data
)
```

テーブルに新しいデータ行を追加します。テーブル内の最大行数は `wandb.Table.MAX_ARTIFACT_ROWS` によって決定されます。

データの長さはテーブル列の長さと一致する必要があります。

### `add_row`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L418-L421)

```python
add_row(
    *row
)
```

非推奨; 代わりに add_data を使用してください。

### `cast`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L315-L371)

```python
cast(
    col_name, dtype, optional=(False)
)
```

列を特定のデータ型にキャストします。

これは通常の Python クラスの1つである場合もあれば、内部の W&B タイプの1つであり、例えば wandb.Image や wandb.Classes のインスタンスのようなサンプルオブジェクトである場合もあります。

| 引数 |  |
| :--- | :--- |
|  `col_name` |  (str) - キャストする列の名前。 |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - 目的の dtype。 |
|  `optional` |  (bool) - 列に None を許可するかどうか。 |

### `get_column`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L838-L861)

```python
get_column(
    name, convert_to=None
)
```

テーブルから列を取得し、オプションで NumPy オブジェクトに変換します。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) - 列の名前 |
|  `convert_to` |  (str, optional) - "numpy": 基礎となるデータを numpy オブジェクトに変換します |

### `get_dataframe`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L872-L878)

```python
get_dataframe()
```

テーブルの `pandas.DataFrame` を返します。

### `get_index`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L863-L870)

```python
get_index()
```

リンクを作成するために他のテーブルで使用する行インデックスの配列を返します。

### `index_ref`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L880-L885)

```python
index_ref(
    index
)
```

テーブル内の行のインデックスの参照を取得します。

### `iterrows`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L674-L688)

```python
iterrows()
```

行ごとにテーブルデータを返し、行のインデックスと関連するデータを表示します。

| Yields |  |
| :--- | :--- |

***

index : int  
行のインデックス。この値を他の W&B テーブルで使用することで、テーブル間の関係が自動的に構築されます  
row : List[any]  
行のデータ。

### `set_fk`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L695-L699)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L690-L693)

```python
set_pk(
    col_name
)
```

| クラス変数 |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |