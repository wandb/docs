# WandbEvalCallback

![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)[GitHubでソースを表示する](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/tables_builder.py#L10-L241)

モデル予測の可視化のためのKerasのコールバックを作成する抽象ベースクラスです。

```python
WandbEvalCallback(
 data_table_columns: List[str],
 pred_table_columns: List[str],
 *args,
 **kwargs
) -> None
```

`on_epoch_end`でモデル予測を可視化するコールバックを作成できます。これらは、分類、物体検出、セグメンテーションなどのタスクで`model.fit()`に渡すことができます。

これを使用するには、このベース コールバック クラスから継承し、`add_ground_truth` メソッドと `add_model_prediction` メソッドを実装します。

ベースクラスは以下を行います。
- 正解をログに記録するための `data_table` と予測のための `pred_table` を初期化します。
- `data_table` にアップロードされたデータは、`pred_table` の参照用データとして使用されます。これは、メモリの使用量を削減するためです。`data_table_ref` は、参照されたデータにアクセスできるリストです。以下の例で使用方法を確認してください。
- W&BアーティファクトとしてテーブルをW&Bにログに記録します。
- 新しい `pred_table` は、エイリアス付きの新しいバージョンとしてログに記録されます。
#### 例:

```
class WandbClfEvalCallback(WandbEvalCallback):
 def __init__(
 self,
 validation_data,
 data_table_columns,
 pred_table_columns
 ):
 super().__init__(
 data_table_columns,
 pred_table_columns
 )

 self.x = validation_data[0]
 self.y = validation_data[1]

 def add_ground_truth(self):
 for idx, (image, label) in enumerate(zip(self.x, self.y)):
 self.data_table.add_data(
 idx,
 wandb.Image(image),
 label
 )

 def add_model_predictions(self, epoch):
 preds = self.model.predict(self.x, verbose=0)
 preds = tf.argmax(preds, axis=-1)

data_table_ref = self.data_table_ref
 table_idxs = data_table_ref.get_index()

 for idx in table_idxs:
 pred = preds[idx]
 self.pred_table.add_data(
 epoch,
 data_table_ref.data[idx][0],
 data_table_ref.data[idx][1],
 data_table_ref.data[idx][2],
 pred
 )

model.fit(
 x,
 y,
 epochs=2,
 validation_data=(x, y),
 callbacks=[
 WandbClfEvalCallback(
 validation_data=(x, y),
 data_table_columns=["idx", "image", "label"],
 pred_table_columns=["epoch", "idx", "image", "label", "pred"])
 ],
)
```

より細かい制御が必要な場合は、`on_train_begin`と`on_epoch_end`メソッドをオーバーライドできます。Nバッチ後のサンプルをログに記録したい場合は、`on_train_batch_end`メソッドを実装できます。
## メソッド

### `add_ground_truth`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/tables_builder.py#L127-L144)

```python
@abc.abstractmethod
add_ground_truth(
 logs: Optional[Dict[str, float]] = None
) -> None
```

`data_table`に正解データを追加します。

`init_data_table`メソッドを使って初期化された`data_table`に検証/トレーニングデータを追加するロジックを書くために、このメソッドを使用します。

#### 例:

```
for idx, data in enumerate(dataloader):
 self.data_table.add_data(
 idx,
 data
 )
```
このメソッドは、`on_train_begin` または同等のフックで一度呼ばれます。

### `add_model_predictions`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/tables_builder.py#L146-L168)

```python
@abc.abstractmethod
add_model_predictions(
 epoch: int,
 logs: Optional[Dict[str, float]] = None
) -> None
```

モデルからの予測を `pred_table` に追加します。

このメソッドを使用して、`init_pred_table` メソッドで初期化された `pred_table` に検証/トレーニングデータのモデル予測を追加するロジックを記述します。

#### 例：

```
# データローダがサンプルをシャッフルしないと仮定します。
for idx, data in enumerate(dataloader):
 preds = model.predict(data)
 self.pred_table.add_data(
 self.data_table_ref.data[idx][0],
 self.data_table_ref.data[idx][1],
 preds
 )
```
このメソッドは、`on_epoch_end` または同等のフックと呼ばれます。

### `init_data_table`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/tables_builder.py#L170-L179)

```python
init_data_table(
 column_names: List[str]
) -> None
```

検証データ用のW&Bテーブルを初期化します。

このメソッドは、`on_train_begin`または同等のフックで呼び出します。これに続いて、テーブルの行または列にデータを追加します。

| 引数 | |
| :--- | :--- |
| column_names (list): W&Bテーブルの列名。 |



### `init_pred_table`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/tables_builder.py#L181-L190)
```python
init_pred_table(
 column_names: List[str]
) -> None
```

モデル評価のためのW&Bテーブルを初期化します。

このメソッドは`on_epoch_end`または同等のフックで呼び出します。これに続いて、テーブルの行または列にデータを追加します。

| Args | |
| :--- | :--- |
| column_names (list): W&Bテーブルの列名。 |


### `log_data_table`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/tables_builder.py#L192-L218)

```python
log_data_table(
 name: str = "val",
 type: str = "dataset",
 table_name: str = "val_data"
) -> None
```
`data_table` をW&Bアーティファクトとしてログし、それに対して `use_artifact` を呼び出します。

これにより、評価テーブルは、すでにアップロードされたデータ（画像、テキスト、スカラーなど）の参照を使用して、再アップロードせずに済みます。

| Args | |
| :--- | :--- |
| name (str): このアーティファクトに対する人間が読める名前で、UI内でこのアーティファクトを識別したり、use_artifact呼び出しで参照したりする際に使用します。（デフォルトは 'val'） type（str）:アーティファクトの種類で、アーティファクトを整理・区別するために使用されます。（デフォルトは 'dataset'） table_name (str): UIに表示されるテーブルの名前（デフォルトは 'val_data'）。|



### `log_pred_table`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/callbacks/tables_builder.py#L220-L241)

```python
log_pred_table(
 type: str = "evaluation",
 table_name: str = "eval_data",
 aliases: Optional[List[str]] = None
) -> None
```

モデルの評価のためのW&Bテーブルをログします。

テーブルは複数回ログされ、新しいバージョンが作成されます。これを使用して
異なる間隔でのモデル比較をインタラクティブに行います。
| Args | |
| :--- | :--- |
| type (str): アーティファクトの種類で、アーティファクトを整理し区別するために使用されます。（デフォルトは 'evaluation'）table_name (str): UIで表示されるテーブルの名前。（デフォルトは 'eval_data'）aliases (List[str]): 予測テーブルのエイリアスのリスト。 |




### `set_model`



```python
set_model(
 model
)
```




### `set_params`



```python
set_params(
 params
)
```