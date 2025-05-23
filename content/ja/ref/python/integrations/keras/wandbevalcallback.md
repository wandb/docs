---
title: WandbEvalCallback
menu:
  reference:
    identifier: ja-ref-python-integrations-keras-wandbevalcallback
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L10-L228 >}}

Keras コールバックをモデル予測の可視化用に構築するための抽象基本クラス。

```python
WandbEvalCallback(
    data_table_columns: List[str],
    pred_table_columns: List[str],
    *args,
    **kwargs
) -> None
```

エポック終了時 (`on_epoch_end`) にモデル予測を可視化するためのコールバックを構築し、分類、オブジェクト検出、セグメンテーションなどのタスク用に `model.fit()` に渡すことができます。

これを使用するには、このベースコールバッククラスから継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装します。

この基本クラスは以下のことを処理します：

- 正解をログするための `data_table` と予測のための `pred_table` を初期化します。
- `data_table` にアップロードされたデータは `pred_table` の参照として使用されます。これはメモリフットプリントを削減するためです。`data_table_ref` は参照されたデータにアクセスするために使用できるリストです。以下の例を見て方法を確認してください。
- W&B にテーブルを W&B Artifacts としてログします。
- 新しい `pred_table` はエイリアスとともに新しいバージョンとしてログされます。

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

より詳細に制御したい場合は、`on_train_begin` と `on_epoch_end` メソッドをオーバーライドできます。N バッチごとにサンプルをログしたい場合は、`on_train_batch_end` メソッドを実装することができます。

## メソッド

### `add_ground_truth`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L117-L131)

```python
@abc.abstractmethod
add_ground_truth(
    logs: Optional[Dict[str, float]] = None
) -> None
```

正解データを `data_table` に追加します。

このメソッドを使用して、`init_data_table` メソッドを使用して初期化された `data_table` にバリデーション/トレーニングデータを追加するロジックを書きます。

#### 例:

```python
for idx, data in enumerate(dataloader):
    self.data_table.add_data(idx, data)
```

このメソッドは、`on_train_begin` または同等のフックで呼び出されます。

### `add_model_predictions`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L133-L155)

```python
@abc.abstractmethod
add_model_predictions(
    epoch: int,
    logs: Optional[Dict[str, float]] = None
) -> None
```

モデルからの予測を `pred_table` に追加します。

このメソッドを使用して、`init_pred_table` メソッドを使用して初期化された `pred_table` にバリデーション/トレーニングデータのモデル予測を追加するロジックを書きます。

#### 例:

```python
# dataloader がサンプルをシャッフルしていないと仮定します。
for idx, data in enumerate(dataloader):
    preds = model.predict(data)
    self.pred_table.add_data(
        self.data_table_ref.data[idx][0],
        self.data_table_ref.data[idx][1],
        preds,
    )
```

このメソッドは、`on_epoch_end` または同等のフックで呼び出されます。

### `init_data_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L157-L166)

```python
init_data_table(
    column_names: List[str]
) -> None
```

バリデーションデータ用の W&B テーブルを初期化します。

このメソッドを `on_train_begin` または同等のフックで呼び出します。これに続いて、テーブルに行または列ごとにデータを追加します。

| 引数 |  |
| :--- | :--- |
|  `column_names` |  (list) W&B テーブルのカラム名です。 |

### `init_pred_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L168-L177)

```python
init_pred_table(
    column_names: List[str]
) -> None
```

モデルの評価用の W&B テーブルを初期化します。

このメソッドを `on_epoch_end` または同等のフックで呼び出します。これに続いて、テーブルに行または列ごとにデータを追加します。

| 引数 |  |
| :--- | :--- |
|  `column_names` |  (list) W&B テーブルのカラム名です。 |

### `log_data_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L179-L205)

```python
log_data_table(
    name: str = "val",
    type: str = "dataset",
    table_name: str = "val_data"
) -> None
```

`data_table` を W&B Artifacts としてログし、`use_artifact` を呼び出します。

これにより、評価テーブルが既にアップロードされたデータ（画像、テキスト、スカラーなど）の参照を再アップロードせずに使用できます。

| 引数 |  |
| :--- | :--- |
|  `name` |  (str) このアーティファクトの人間が読める名前で、UIでこのアーティファクトを識別したり、use_artifact呼び出しで参照したりする方法です。（デフォルトは 'val'） |
|  `type` |  (str) アーティファクトのタイプで、アーティファクトを整理し区別するために使用されます。（デフォルトは 'dataset'） |
|  `table_name` |  (str) UIで表示されるテーブルの名前です。（デフォルトは 'val_data'） |

### `log_pred_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L207-L228)

```python
log_pred_table(
    type: str = "evaluation",
    table_name: str = "eval_data",
    aliases: Optional[List[str]] = None
) -> None
```

モデルの評価用の W&B テーブルをログします。

テーブルは新しいバージョンを作成しながら複数回ログされます。これを使用して、異なる間隔でモデルをインタラクティブに比較します。

| 引数 |  |
| :--- | :--- |
|  `type` |  (str) アーティファクトのタイプで、アーティファクトを整理し区別するために使用されます。（デフォルトは 'evaluation'） |
|  `table_name` |  (str) UIで表示されるテーブルの名前です。（デフォルトは 'eval_data'） |
|  `aliases` |  (List[str]) 予測テーブルのエイリアスのリストです。 |

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