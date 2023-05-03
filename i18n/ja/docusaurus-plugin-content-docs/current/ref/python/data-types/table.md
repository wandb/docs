# テーブル

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L153-L936)

Tableクラスは、表形式のデータを表示・分析するために使用されます。

```python
Table(
 columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
 allow_mixed_types=(False)
)
```

従来のスプレッドシートとは異なり、Tableは様々なタイプのデータに対応しています：スカラー値、文字列、numpy配列、そしてほとんどの`wandb.data_types.Media`のサブクラスも対応しています。
これにより、`Images`、`Video`、`Audio`などのリッチでアノテーションされたメディアを、他の従来のスカラー値と並べてTableに埋め込むことができます。

このクラスは、UIのTable Visualizerを生成する主要なクラスです：https://docs.wandb.ai/guides/data-vis/tables.

Tableは、`data`または`dataframe` パラメータを使用して初期データを含めて作成することができます：
以下は、Markdownテキストのチャンクです。これを日本語に翻訳してください。それ以外のことは何も言わず、翻訳されたテキストのみを返してください。テキスト：

```python
import pandas as pd
import wandb

data = {"users": ["geoff", "juergen", "ada"], "feature_01": [1, 117, 42]}
df = pd.DataFrame(data)

tbl = wandb.Table(data=df)
assert all(tbl.get_column("users") == df["users"])
assert all(tbl.get_column("feature_01") == df["feature_01"])
```

さらに、ユーザーは、
`add_data`、`add_column`、および `add_computed_column` 関数を使用して、
行、列、および他の列のデータに基づいて計算された列をそれぞれ追加することにより、Tablesにデータをインクリメンタルに追加できます。

```python
import wandb

tbl = wandb.Table(columns=["user"])

users = ["geoff", "juergen", "ada"]

[tbl.add_data(user) for user in users]
assert tbl.get_column("user") == users


def get_user_name_length(index, row):
 return {"feature_01": len(row["user"])}
以下のMarkdownテキストを日本語に翻訳してください。それ以外のことは言わず、翻訳されたテキストだけを返してください。テキスト:

```python
tbl.add_computed_columns(get_user_name_length)
assert tbl.get_column("feature_01") == [5, 7, 3]
```

テーブルは、`run.log({"my_table": table})`を使用して直接runsにログに記録することができます。
また、`artifact.add(table, "my_table")`を使用してアーティファクトに追加できます：

```python
import numpy as np
import wandb

wandb.init()

tbl = wandb.Table(columns=["image", "label"])

images = np.random.randint(0, 255, [2, 100, 100, 3], dtype=np.uint8)
labels = ["panda", "gibbon"]
[tbl.add_data(wandb.Image(image), label) for image, label in zip(images, labels)]

wandb.log({"classifier_out": tbl})
```

上記のように直接runsに追加されたテーブルは、ワークスペース内の対応するTable Visualizerを生成し、さらなる分析やレポートへのエクスポートに使用できます。

アーティファクトに追加されたテーブルは、Artifactタブで表示でき、アーティファクトブラウザ内で同等のTable Visualizerがレンダリングされます。

テーブルは、各列の値が同じタイプであることを期待しています。デフォルトでは、列はオプションの値をサポートしますが、混在した値はサポートしません。絶対にタイプを混ぜる必要がある場合は、`allow_mixed_types`フラグを有効にして、データの型チェックを無効にすることができます。これにより、一貫した型付けが欠けているため、一部のテーブル分析機能が無効になります。
以下は、Markdownテキストのチャンクを翻訳してください。日本語に翻訳し、翻訳されたテキストのみを返してください。他のことは何も言わないでください。テキスト：

| 引数 |  |
| :--- | :--- |
| `columns` | (List[str]) テーブルの列名。デフォルトでは["Input", "Output", "Expected"]。 |
| `data` | (List[List[any]]) 値の2次元の行指向配列。 |
| `dataframe` | (pandas.DataFrame) テーブルを作成するために使用されるDataFrameオブジェクト。設定されている場合、`data` と `columns` の引数は無視されます。 |
| `optional` | (Union[bool,List[bool]]) `None`の値が許可されるかどうかを決定します。デフォルトではTrue - 単一のbool値の場合、構築時に指定されたすべての列にオプションが強制されます - 列と同じ長さであるべきbool値のリストの場合、オプションが各列に適用されます。 |
| `allow_mixed_types` | (bool) 列に混在した型が許可されているかどうかを決定します（型検証が無効になります）。デフォルトでは False |



## メソッド

### `add_column`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L832-L871)

```python
add_column(
 name, data, optional=(False)
)
```

テーブルにデータの列を追加します。

| 引数 |  |
| :--- | :--- |
| `name` | (str) - 列の一意の名前|
| `data` | (list | np.array) - 同種のデータの列 |
| `optional` | (bool) - nullのような値が許可されているかどうか |
### `add_computed_columns`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L914-L936)

```python
add_computed_columns(
 fn
)
```

既存のデータに基づいて、1つまたは複数の計算された列を追加します。

| 引数 | |
| :--- | :--- |
| `fn` | `ndx`（int）と`row`（dict）という1つまたは2つのパラメータを受け取り、その行の新しい列を表す辞書を返すことが期待される関数。`ndx`は、行のインデックスを表す整数です。`include_ndx`が`True`に設定されている場合にのみ含まれます。`row`は、既存の列によってキー化された辞書です |



### `add_data`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L458-L491)

```python
add_data(
 *data
)
```
表にデータの行を追加します。

引数の長さは、カラムの長さと一致する必要があります。

### `add_row`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L453-L456)

```python
add_row(
 *row
)
```

非推奨：代わりに add_data を使用してください。

### `cast`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L352-L406)

```python
cast(
 col_name, dtype, optional=(False)
)
```
特定の型に列をキャストする。

| 引数 | |
| :--- | :--- |
| `col_name` | (str) - キャストする列の名前 |
| `dtype` | (class, wandb.wandb_sdk.interface._dtypes.Type, any) - 対象のdtype。通常のPythonクラス、内部WBタイプ、または例オブジェクト（例： wandb.Image や wandb.Classes のインスタンス）のいずれかにできます。 |
| `optional` | (bool) - その列がNoneを許可するかどうか |



### `get_column`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L873-L896)

```python
get_column(
 name, convert_to=None
)
```

テーブルからデータ列を取得します。

| 引数 | |
| :--- | :--- |
| `name` | (str) - 列の名前 |
| `convert_to` | (str, optional) - "numpy": 基礎となるデータをnumpyオブジェクトに変換する |
### `get_index`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L898-L905)

```python
get_index()
```

リンクを作成するための他のテーブルで使用する行インデックスの配列を返します。


### `index_ref`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L907-L912)

```python
index_ref(
 index
)
```

テーブル内の特定の行インデックスへの参照を取得します。
### `iterrows`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L709-L723)

```python
iterrows()
```

行を（ndx、行）としてイテレートします。

| Yields | |
| :--- | :--- |

------
index : int
 行のインデックス。この値を他のWandBテーブルで使用すると、テーブル間の関係が自動的に構築されます。
row : List[any]
 行のデータ。

### `set_fk`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L730-L734)
以下は、Markdownテキストのチャンクを翻訳してください。日本語に翻訳し、それ以外のことは何も言わずに翻訳されたテキストを返してください。テキスト：

```python
set_fk(
 col_name, table, table_col
)
```




### `set_pk`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/data_types.py#L725-L728)

```python
set_pk(
 col_name
)
```








| クラス変数 | |
| :--- | :--- |
| `MAX_ARTIFACT_ROWS` | `200000` |
| `MAX_ROWS` | `10000` |