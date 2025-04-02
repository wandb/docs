---
title: Table
menu:
  reference:
    identifier: ja-ref-python-data-types-table
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L183-L909 >}}

テーブル形式のデータを表示および分析するために使用される Table クラス。

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

従来の表計算ソフトとは異なり、Tables は、スカラー値、文字列、NumPy 配列、および `wandb.data_types.Media` のほとんどのサブクラスなど、多数の種類のデータをサポートしています。
これは、`Images`、`Video`、`Audio`、およびその他の種類の豊富なアノテーション付きメディアを、他の従来のスカラー値とともに Tables に直接埋め込むことができることを意味します。

このクラスは、UI で Table Visualizer を生成するために使用される主要なクラスです: https://docs.wandb.ai/guides/data-vis/tables 。

| Args |  |
| :--- | :--- |
|  `columns` |  (List[str]) テーブル内の列の名前。デフォルトは ["Input", "Output", "Expected"] です。 |
|  `data` |  (List[List[any]]) 値の 2D 行指向配列。 |
|  `dataframe` |  (pandas.DataFrame) テーブルの作成に使用される DataFrame オブジェクト。設定すると、`data` および `columns` 引数は無視されます。 |
|  `optional` |  (Union[bool,List[bool]]) `None` 値を許可するかどうかを決定します。デフォルトは True です - 単一の bool 値の場合、構築時に指定されたすべての列に対して optionality が適用されます - bool 値のリストの場合、optionality は各列に適用されます - `columns` と同じ長さである必要があります。bool 値のリストは、それぞれの列に適用されます。 |
|  `allow_mixed_types` |  (bool) 列で混合型を使用できるかどうかを決定します (型検証を無効にします)。デフォルトは False です。 |

## メソッド

### `add_column`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L797-L836)

```python
add_column(
    name, data, optional=(False)
)
```

データの列をテーブルに追加します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) - 列の一意の名前 |
|  `data` |  (list | np.array) - 同種のデータの列 |
|  `optional` |  (bool) - null のような値が許可されているかどうか |

### `add_computed_columns`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L887-L909)

```python
add_computed_columns(
    fn
)
```

既存のデータに基づいて、1 つまたは複数の計算列を追加します。

| Args |  |
| :--- | :--- |
|  `fn` |  1 つまたは 2 つの パラメータ (ndx (int) および row (辞書)) を受け入れる関数。この関数は、その行の新しい列を表す辞書を返すことが想定されており、新しい列名でキーが設定されています。`ndx` は、行のインデックスを表す整数です。`include_ndx` が `True` に設定されている場合にのみ含まれます。`row` は、既存の列でキーが設定された辞書です。 |

### `add_data`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L423-L456)

```python
add_data(
    *data
)
```

テーブルにデータの新しい行を追加します。テーブル内の最大行数は、`wandb.Table.MAX_ARTIFACT_ROWS` によって決定されます。

データの長さは、テーブル列の長さと一致する必要があります。

### `add_row`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L418-L421)

```python
add_row(
    *row
)
```

非推奨。代わりに add_data を使用してください。

### `cast`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L315-L371)

```python
cast(
    col_name, dtype, optional=(False)
)
```

列を特定のデータ型にキャストします。

これは、通常の Python クラス、内部の W&B 型、または wandb.Image や wandb.Classes のインスタンスのようなサンプル オブジェクトのいずれかになります。

| Args |  |
| :--- | :--- |
|  `col_name` |  (str) - キャストする列の名前。 |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - ターゲットの dtype。 |
|  `optional` |  (bool) - 列で None を許可する必要があるかどうか。 |

### `get_column`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L838-L861)

```python
get_column(
    name, convert_to=None
)
```

テーブルから列を取得し、必要に応じて NumPy オブジェクトに変換します。

| Args |  |
| :--- | :--- |
|  `name` |  (str) - 列の名前 |
|  `convert_to` |  (str, optional) - "numpy": 基になるデータを numpy オブジェクトに変換します |

### `get_dataframe`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L872-L878)

```python
get_dataframe()
```

テーブルの `pandas.DataFrame` を返します。

### `get_index`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L863-L870)

```python
get_index()
```

他のテーブルでリンクを作成するために使用する行インデックスの配列を返します。

### `index_ref`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L880-L885)

```python
index_ref(
    index
)
```

テーブル内の行のインデックスの参照を取得します。

### `iterrows`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/table.py#L674-L688)

```python
iterrows()
```

行ごとのテーブルデータを返し、行のインデックスと関連データを表示します。

| Yields |  |
| :--- | :--- |

***

index : int
行のインデックス。この値を他の W&B テーブルで使用すると、テーブル間の関係が自動的に構築されます
row : List[any]
行のデータ。

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
