---
title: WandbEvalCallback
menu:
  reference:
    identifier: ja-ref-python-integrations-keras-wandbevalcallback
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L10-L228 >}}

モデルの予測の可視化のために Keras の callback を構築するための抽象基底クラス。

```python
WandbEvalCallback(
    data_table_columns: List[str],
    pred_table_columns: List[str],
    *args,
    **kwargs
) -> None
```

分類、オブジェクト検出、セグメンテーションなどのタスクのために、`model.fit()` に渡すことができる `on_epoch_end` でモデルの予測を可視化するための callback を構築できます。

これを使用するには、この基底 callback クラスを継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装します。

基底クラスは、以下を処理します。

- 正解をログするための `data_table` と、予測のための `pred_table` を初期化します。
- `data_table` にアップロードされたデータは、`pred_table` の参照として使用されます。これは、メモリフットプリントを削減するためです。`data_table_ref` は、参照されたデータにアクセスするために使用できるリストです。その方法については、以下の例を確認してください。
- テーブルを W&B に W&B Artifacts として記録します。
- 新しい `pred_table` はそれぞれ、エイリアスを持つ新しいバージョンとしてログに記録されます。

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

よりきめ細かい制御を行うには、`on_train_begin` メソッドと `on_epoch_end` メソッドをオーバーライドできます。N 個のバッチ処理後にサンプルをログに記録する場合は、`on_train_batch_end` メソッドを実装できます。

## メソッド

### `add_ground_truth`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L117-L131)

```python
@abc.abstractmethod
add_ground_truth(
    logs: Optional[Dict[str, float]] = None
) -> None
```

正解 データを `data_table` に追加します。

このメソッドを使用して、`init_data_table` メソッドを使用して初期化された `data_table` に、検証/トレーニングデータを追加するロジックを記述します。

#### 例:

```python
for idx, data in enumerate(dataloader):
    self.data_table.add_data(idx, data)
```

このメソッドは、`on_train_begin` または同等の hook で 1 回呼び出されます。

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

このメソッドを使用して、`init_pred_table` メソッドを使用して初期化された `pred_table` に、検証/トレーニングデータのモデル予測を追加するロジックを記述します。

#### 例:

```python
# データローダーがサンプルをシャッフルしないと仮定します。
for idx, data in enumerate(dataloader):
    preds = model.predict(data)
    self.pred_table.add_data(
        self.data_table_ref.data[idx][0],
        self.data_table_ref.data[idx][1],
        preds,
    )
```

このメソッドは、`on_epoch_end` または同等の hook で呼び出されます。

### `init_data_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L157-L166)

```python
init_data_table(
    column_names: List[str]
) -> None
```

検証データ 用の W&B Tables を初期化します。

このメソッドを `on_train_begin` または同等の hook で呼び出します。これに続いて、テーブルに行ごとまたは列ごとにデータを追加します。

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) W&B Tables の列名。 |

### `init_pred_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L168-L177)

```python
init_pred_table(
    column_names: List[str]
) -> None
```

モデル評価用の W&B Tables を初期化します。

このメソッドを `on_epoch_end` または同等の hook で呼び出します。これに続いて、テーブルに行ごとまたは列ごとにデータを追加します。

| Args |  |
| :--- | :--- |
|  `column_names` |  (list) W&B Tables の列名。 |

### `log_data_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L179-L205)

```python
log_data_table(
    name: str = "val",
    type: str = "dataset",
    table_name: str = "val_data"
) -> None
```

`data_table` を W&B artifact としてログに記録し、それに対して `use_artifact` を呼び出します。

これにより、評価テーブルは、すでにアップロードされたデータ (画像、テキスト、スカラーなど) の参照を、再アップロードせずに使用できます。

| Args |  |
| :--- | :--- |
|  `name` |  (str) この Artifact の人間が読める名前。UI でこの Artifact を識別したり、use_artifact 呼び出しで参照したりする方法です。(デフォルトは 'val') |
|  `type` |  (str) Artifact のタイプ。Artifact を整理および区別するために使用されます。(デフォルトは 'dataset') |
|  `table_name` |  (str) UI に表示されるテーブルの名前。(デフォルトは 'val_data')。 |

### `log_pred_table`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/tables_builder.py#L207-L228)

```python
log_pred_table(
    type: str = "evaluation",
    table_name: str = "eval_data",
    aliases: Optional[List[str]] = None
) -> None
```

モデル評価用の W&B Tables をログに記録します。

テーブルは複数回ログに記録され、新しいバージョンが作成されます。これを使用して、さまざまな間隔でモデルをインタラクティブに比較します。

| Args |  |
| :--- | :--- |
|  `type` |  (str) Artifact のタイプ。Artifact を整理および区別するために使用されます。(デフォルトは 'evaluation') |
|  `table_name` |  (str) UI に表示されるテーブルの名前。(デフォルトは 'eval_data') |
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