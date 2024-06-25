
# WandbEvalCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/tables_builder.py#L10-L226' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Kerasのコールバックを作成するための抽象基本クラスで、モデル予測の可視化に使用します。

```python
WandbEvalCallback(
    data_table_columns: List[str],
    pred_table_columns: List[str],
    *args,
    **kwargs
) -> None
```

分類、オブジェクト検出、セグメンテーションなどのタスクで、モデル予測を `on_epoch_end` で可視化するためのコールバックを作成し、`model.fit()` に渡すことができます。

これを使用するには、この基本コールバッククラスを継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装します。

基本クラスは以下の処理を行います：

- 正解を記録するための `data_table` と予測を記録するための `pred_table` の初期化。
- `data_table` にアップロードされたデータは `pred_table` の参照として使用されます。これによりメモリのフットプリントが削減されます。`data_table_ref` は参照されたデータにアクセスするためのリストです。以下の例を参照してください。
- テーブルを W&B としてログし、W&B Artifacts として記録します。
- 各新しい `pred_table` はエイリアスとともに新しいバージョンとして記録されます。

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
    エポック数=2,
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

より細かな制御が必要な場合は、`on_train_begin` と `on_epoch_end` メソッドをオーバーライドすることもできます。Nバッチ後にサンプルをログしたい場合は、`on_train_batch_end` メソッドを実装できます。

## メソッド

### `add_ground_truth`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/tables_builder.py#L117-L131)

```python
@abc.abstractmethod
add_ground_truth(
    logs: Optional[Dict[str, float]] = None
) -> None
```

`data_table` に正解データを追加します。

このメソッドを使用して、`init_data_table` メソッドを使用して初期化された `data_table` に検証/トレーニングデータを追加するロジックを書きます。

#### 例:

```python
for idx, data in enumerate(dataloader):
    self.data_table.add_data(idx, data)
```

このメソッドは一度 `on_train_begin` または同等のフックで呼び出されます。

### `add_model_predictions`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/tables_builder.py#L133-L153)

```python
@abc.abstractmethod
add_model_predictions(
    epoch: int,
    logs: Optional[Dict[str, float]] = None
) -> None
```

`pred_table` にモデルの予測を追加します。

このメソッドを使用して、`init_pred_table` メソッドを使用して初期化された `pred_table` に検証/トレーニングデータのモデル予測を追加するロジックを書きます。

#### 例:

```python
# サンプルがシャッフルされていないと仮定します。
for idx, data in enumerate(dataloader):
    preds = model.predict(data)
    self.pred_table.add_data(
        self.data_table_ref.data[idx][0], self.data_table_ref.data[idx][1], preds
    )
```

このメソッドは `on_epoch_end` または同等のフックで呼び出されます。

### `init_data_table`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/tables_builder.py#L155-L164)

```python
init_data_table(
    column_names: List[str]
) -> None
```

W&B Tables を検証データ用に初期化します。

このメソッドを `on_train_begin` または同等のフックで呼び出します。その後、行または列ごとにテーブルにデータを追加します。

| Args |  |
| :--- | :--- |
|  `column_names` |  (リスト) W&B Tables の列名。 |

### `init_pred_table`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/tables_builder.py#L166-L175)

```python
init_pred_table(
    column_names: List[str]
) -> None
```

モデルの評価用のW&B Tablesを初期化します。

このメソッドを `on_epoch_end` または同等のフックで呼び出します。その後、行または列ごとにテーブルにデータを追加します。

| Args |  |
| :--- | :--- |
|  `column_names` |  (リスト) W&B Tables の列名。 |

### `log_data_table`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/tables_builder.py#L177-L203)

```python
log_data_table(
    name: str = "val",
    type: str = "dataset",
    table_name: str = "val_data"
) -> None
```

`data_table` を W&B アーティファクトとして記録し、`use_artifact` を呼び出します。

これにより、評価テーブルは既にアップロードされたデータ（画像、テキスト、スカラーなど）の参照を使用し、再アップロードすることなく利用できます。

| Args |  |
| :--- | :--- |
|  `name` |  (str) このアーティファクトの人間が読みやすい名前。UI でこのアーティファクトを識別したり `use_artifact` 呼び出しで参照します。(デフォルトは 'val') |
|  `type` |  (str) アーティファクトのタイプで、アーティファクトを整理および区別するために使用します。(デフォルトは 'dataset') |
|  `table_name` |  (str) UI で表示されるテーブルの名前。(デフォルトは 'val_data'). |

### `log_pred_table`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/tables_builder.py#L205-L226)

```python
log_pred_table(
    type: str = "evaluation",
    table_name: str = "eval_data",
    aliases: Optional[List[str]] = None
) -> None
```

モデル評価用のW&B Tablesを記録します。

このテーブルは複数回ログされ、新しいバージョンが作成されます。これを使用して、異なる間隔でのモデルを対話的に比較します。

| Args |  |
| :--- | :--- |
|  `type` |  (str) アーティファクトのタイプで、アーティファクトを整理および区別するために使用します。(デフォルトは 'evaluation') |
|  `table_name` |  (str) UI で表示されるテーブルの名前。(デフォルトは 'eval_data') |
|  `aliases` |  (リスト) 予測テーブルのエイリアスのリスト。 |

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