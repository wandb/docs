# WandbCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L291-L1080' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

`WandbCallback` は keras と wandb を自動的に統合します。

```python
WandbCallback(
    monitor="val_loss", verbose=0, mode="auto", save_weights_only=(False),
    log_weights=(False), log_gradients=(False), save_model=(True),
    training_data=None, validation_data=None, labels=None, predictions=36,
    generator=None, input_type=None, output_type=None, log_evaluation=(False),
    validation_steps=None, class_colors=None, log_batch_frequency=None,
    log_best_prefix="best_", save_graph=(True), validation_indexes=None,
    validation_row_processor=None, prediction_row_processor=None,
    infer_missing_processors=(True), log_evaluation_frequency=0,
    compute_flops=(False), **kwargs
)
```

#### 例:

```python
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback()],
)
```

`WandbCallback` は keras によって収集されたメトリクスから、いかなる履歴データも自動的にログします: 損失および `keras_model.compile()` に渡されたもの全て。

`WandbCallback` は "best" training ステップに関連付けられた run のサマリーメトリクスを設定します。"best" は `monitor` および `mode` 属性によって定義され、デフォルトでは、最小の `val_loss` を持つエポックになります。`WandbCallback` はデフォルトで最も良い `epoch` に関連付けられたモデルを保存します。

`WandbCallback` はオプションで勾配およびパラメータのヒストグラムをログすることができます。

`WandbCallback` はオプションでトレーニングおよび検証データを保存し、wandb で可視化することができます。

| 引数 |  |
| :--- | :--- |
|  `monitor` |  (str) 監視するメトリクスの名前。デフォルトは `val_loss`。 |
|  `mode` |  (str) 次のいずれか: {`auto`, `min`, `max`}. `min` - 監視値が最小化された時、モデルを保存します。`max` - 監視値が最大化された時、モデルを保存します。`auto` - モデルを保存する時を推測しようとします（デフォルト）。|
|  `save_model` |  True - すべての過去のエポックを上回る監視値がある場合、モデルを保存します。False - モデルを保存しません。 |
|  `save_graph` |  (boolean) True の場合、モデルのグラフを wandb に保存します（デフォルトは True）。 |
|  `save_weights_only` |  (boolean) True の場合、モデルの重みだけを保存します（`model.save_weights(filepath)`）、そうでなければ完全なモデルが保存されます（`model.save(filepath)`）。 |
|  `log_weights` |  (boolean) True の場合、モデルの層の重みのヒストグラムを保存します。 |
|  `log_gradients` |  (boolean) True の場合、トレーニングの勾配のヒストグラムをログします。 |
|  `training_data` |  (tuple) `model.fit` に渡された `(X, y)` と同じ形式。これは勾配の計算に必要であり、`log_gradients` が True の場合、必須です。 |
|  `validation_data` |  (tuple) `model.fit` に渡された `(X, y)` と同じ形式。wandb で可視化するためのデータセット。これが設定されている場合、各エポック、wandb は少数の予測を行い、後で可視化するために結果を保存します。画像データを扱っている場合、正しくログするために `input_type` および `output_type` を設定してください。 |
|  `generator` |  (generator) wandb で可視化するための検証データを返すジェネレータ。このジェネレータは `(X, y)` のタプルを返すべきです。特定のデータ例を可視化するには、`validate_data` または `generator` のいずれかが設定されている必要があります。画像データを扱っている場合、正しくログするために `input_type` および `output_type` を設定してください。 |
|  `validation_steps` |  (int) `validation_data` がジェネレータの場合、完全な検証セットのためにジェネレータを実行するステップ数。 |
|  `labels` |  (list) wandb でデータを可視化している場合、このラベルのリストは数値出力を理解可能な文字列に変換します。複数クラス分類器を構築している場合に有効です。バイナリ分類器を作成している場合、2つのラベル ["false のラベル", "true のラベル"] を渡すことができます。`validate_data` とジェネレータの両方が false の場合、これは何も行いません。 |
|  `predictions` |  (int) 各エポックで可視化のために予測する数、最大は 100。 |
|  `input_type` |  (string) 可視化を助けるためのモデル入力のタイプ。次のいずれか: (`image`, `images`, `segmentation_mask`, `auto`)。 |
|  `output_type` |  (string) 可視化を助けるためのモデル出力のタイプ。次のいずれか: (`image`, `images`, `segmentation_mask`, `label`)。 |
|  `log_evaluation` |  (boolean) True の場合、各エポックで検証データとモデルの予測を含むテーブルを保存します。詳細は `validation_indexes`、`validation_row_processor`、および `output_row_processor` を参照してください。 |
|  `class_colors` |  ([float, float, float]) 入力または出力がセグメンテーションマスクである場合、各クラスの rgb タプル（範囲 0-1）を含む配列。 |
|  `log_batch_frequency` |  (integer) None の場合、コールバックは各エポックでログします。整数に設定された場合、`log_batch_frequency` バッチごとにトレーニングメトリクスをログします。 |
|  `log_best_prefix` |  (string) None の場合、追加のサマリーメトリクスは保存されません。文字列に設定された場合、監視されたメトリクスとエポックはこの値で前置され、サマリーメトリクスとして保存されます。 |
|  `validation_indexes` |  ([wandb.data_types._TableLinkMixin]) 各検証例に関連付けられたインデックスキーの順序付きリスト。`log_evaluation` が True で `validation_indexes` が提供されている場合、検証データのテーブルは作成されず、各予測は `TableLinkMixin` によって表される行に関連付けられます。通常は `Table.get_index()` を使用して行キーのリストを取得します。 |
|  `validation_row_processor` |  (Callable) 検証データに適用する関数。通常データを可視化するために使用されます。この関数は `ndx`（int）および `row`（dict）を受け取ります。モデルが単一の入力を持つ場合、`row["input"]` はその行の入力データとなります。そうでない場合、入力スロットの名前に基づいてキーが設定されます。fit 関数が単一のターゲットを取る場合、`row["target"]` はその行のターゲットデータとなります。そうでない場合、出力スロットの名前に基づいてキーが設定されます。例えば、入力データが単一の ndarray であり、データを画像として可視化したい場合、`lambda ndx, row: {"img": wandb.Image(row["input"])}` をプロセッサとして提供できます。`log_evaluation` が False または `validation_indexes` が存在する場合は無視されます。 |
|  `output_row_processor` |  (Callable) `validation_row_processor` と同様ですが、モデルの出力に適用されます。`row["output"]` はモデル出力の結果を含みます。 |
|  `infer_missing_processors` |  (bool) `validation_row_processor` および `output_row_processor` が欠けている場合に推論するかどうかを決定します。デフォルトは True です。`labels` が提供されている場合、適切な分類タイプのプロセッサを推論しようとします。 |
|  `log_evaluation_frequency` |  (int) 評価結果をログする頻度を決定します。デフォルトは 0（トレーニングの最後にのみログします）。1 に設定すると各エポックごとにログし、2 に設定すると2 エポックごとにログします。`log_evaluation` が False の場合、効果はありません。 |
|  `compute_flops` |  (bool) Keras の Sequential または Functional モデルの FLOPs を GigaFLOPs 単位で計算します。 |

## メソッド

### `get_flops`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L1034-L1080)

```python
get_flops() -> float
```

推論モードでの tf.keras.Model または tf.keras.Sequential モデルの FLOPS [GFLOPs] を計算します。

内部では tf.compat.v1.profiler を使用します。

### `set_model`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L554-L563)

```python
set_model(
    model
)
```

### `set_params`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/keras.py#L551-L552)

```python
set_params(
    params
)
```