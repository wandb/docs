# WandbEvalCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L10-L226' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Kerasのコールバックを構築してモデル予測の可視化を行うための抽象基底クラス。

```python
WandbEvalCallback(
    data_table_columns: List[str],
    pred_table_columns: List[str],
    *args,
    **kwargs
) -> None
```

`on_epoch_end` にモデル予測を可視化するコールバックを構築して、分類、オブジェクト検出、セグメンテーションなどのタスクのために `model.fit()` に渡すことができます。

これを使用するには、この基底コールバッククラスを継承し、`add_ground_truth` と `add_model_prediction` のメソッドを実装します。

基底クラスは以下を処理します：

- 正解をログするための `data_table` と予測をログするための `pred_table` を初期化。
- `data_table` にアップロードされたデータは `pred_table` の参照として使用されます。これはメモリの消費を減らすためです。`data_table_ref` は参照データにアクセスするために使用できるリストです。以下の例を確認してください。
- テーブルをW&Bとしてログし、W&B Artifactsとして保存。
- 各新しい `pred_table` はエイリアスとともに新しいバージョンとしてログされます。

#### 例:

```python
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(self, validation_data, data_table_columns, pred_table_columns):
        super().__init__(data_table_columns, pred_table_columns)

        self.x = validation_data[0]
        self.y = validation_data[1]

    def add_ground_truth(self):
        for idx, (image, label) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(image), label)

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
                pred,
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
            pred_table_columns=["epoch", "idx", "image", "label", "pred"],
        )
    ],
)
```

より細かい制御を行うために、`on_train_begin` と `on_epoch_end` メソッドをオーバーライドすることができます。N バッチ後にサンプルをログしたい場合は、`on_train_batch_end` メソッドを実装できます。

## メソッド

### `add_ground_truth`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L117-L131)

```python
@abc.abstractmethod
add_ground_truth(
    logs: Optional[Dict[str, float]] = None
) -> None
```

正解データを `data_table` に追加します。

このメソッドを使用して、`init_data_table` メソッドを使用して初期化された `data_table` に検証/トレーニングデータを追加するロジックを記述します。

#### 例:

```python
for idx, data in enumerate(dataloader):
    self.data_table.add_data(idx, data)
```

このメソッドは `on_train_begin` または同等のフックが呼び出された時に一度だけ呼ばれます。

### `add_model_predictions`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L133-L153)

```python
@abc.abstractmethod
add_model_predictions(
    epoch: int,
    logs: Optional[Dict[str, float]] = None
) -> None
```

モデルの予測を `pred_table` に追加します。

このメソッドを使用して、`init_pred_table` メソッドを使用して初期化された `pred_table` に検証/トレーニングデータのモデル予測を追加するためのロジックを書きます。

#### 例:

```python
# Assuming the dataloader is not shuffling the samples.
for idx, data in enumerate(dataloader):
    preds = model.predict(data)
    self.pred_table.add_data(
        self.data_table_ref.data[idx][0], self.data_table_ref.data[idx][1], preds
    )
```

このメソッドは `on_epoch_end` または同等のフックが呼び出された時に実行されます。

### `init_data_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L155-L164)

```python
init_data_table(
    column_names: List[str]
) -> None
```

検証データのための W&B テーブルを初期化します。

このメソッドは `on_train_begin` または同等のフックが呼び出された時に実行され、次に行または列ごとにデータをテーブルに追加します。

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) W&B テーブルの列名。 |

### `init_pred_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L166-L175)

```python
init_pred_table(
    column_names: List[str]
) -> None
```

モデル評価のための W&B テーブルを初期化します。

このメソッドは `on_epoch_end` または同等のフックが呼び出された時に実行され、次に行または列ごとにデータをテーブルに追加します。

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) W&B テーブルの列名。 |

### `log_data_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L177-L203)

```python
log_data_table(
    name: str = "val",
    type: str = "dataset",
    table_name: str = "val_data"
) -> None
```

`data_table` を W&B アーティファクトとしてログし、それに `use_artifact` を呼び出します。

これにより、評価テーブルはアップロード済みのデータ（画像、テキスト、スカラーなど）の参照を再アップロードせずに使用できます。

| Args |  |
| :--- | :--- |
|  `name` |  (str) このアーティファクトの人が読みやすい名前。これはUIでこのアーティファクトを識別するためや、`use_artifact` 呼び出しで参照するためのものです。（デフォルトは 'val'） |
|  `type` |  (str) アーティファクトの種類。アーティファクトの整理と区別に使用されます。（デフォルトは 'dataset'） |
|  `table_name` |  (str) UIに表示されるテーブルの名前。（デフォルトは 'val_data'）。 |

### `log_pred_table`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/tables_builder.py#L205-L226)

```python
log_pred_table(
    type: str = "evaluation",
    table_name: str = "eval_data",
    aliases: Optional[List[str]] = None
) -> None
```

モデル評価のための W&B テーブルをログします。

このテーブルは新しいバージョンを作成するたびに複数回ログされます。これを使い、異なるインターバルでモデルをインタラクティブに比較します。

| Args |  |
| :--- | :--- |
|  `type` |  (str) アーティファクトの種類。アーティファクトの整理と区別に使用されます。（デフォルトは 'evaluation'） |
|  `table_name` |  (str) UIに表示されるテーブルの名前。（デフォルトは 'eval_data'） |
|  `aliases` |  (List[str]) 予測テーブルのエイリアスのリスト。 |

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
