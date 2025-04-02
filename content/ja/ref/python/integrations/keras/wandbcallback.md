---
title: WandbCallback
menu:
  reference:
    identifier: ja-ref-python-integrations-keras-wandbcallback
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L291-L1091 >}}

`WandbCallback` は、keras と wandb を自動的に インテグレーション します。

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

`WandbCallback` は、keras によって収集されたすべての メトリクス (loss や `keras_model.compile()` に渡されたもの) から履歴 データを自動的に ログ します。

`WandbCallback` は、"best" の トレーニング ステップに関連付けられた run の概要 メトリクス を設定します。ここで、"best" は `monitor` および `mode` 属性によって定義されます。これはデフォルトで最小の `val_loss` を持つ エポック になります。`WandbCallback` は、デフォルトで best の `epoch` に関連付けられた model を保存します。

`WandbCallback` は、オプションで 勾配 と パラメータ のヒストグラムを ログ できます。

`WandbCallback` は、オプションで wandb が 可視化 するための トレーニング データ と 検証データ を保存できます。

|  Arg |   |
| :--- | :--- |
|  `monitor` |  (str) 監視する メトリクス の名前。デフォルトは `val_loss` です。 |
|  `mode` |  (str) {`auto`, `min`, `max`} のいずれか。`min` - monitor が最小化されたときに model を保存します `max` - monitor が最大化されたときに model を保存します `auto` - model を保存するタイミングを推測しようとします (デフォルト)。 |
|  `save_model` |  True - monitor が以前のすべての エポック より優れている場合に model を保存します False - model を保存しません |
|  `save_graph` |  (boolean) True の場合、model グラフを wandb に保存します (デフォルトは True)。 |
|  `save_weights_only` |  (boolean) True の場合、model の重みのみが保存されます (`model.save_weights(filepath)`)。それ以外の場合は、完全な model が保存されます (`model.save(filepath)`)。 |
|  `log_weights` |  (boolean) True の場合、model のレイヤーの重みのヒストグラムを保存します。 |
|  `log_gradients` |  (boolean) True の場合、トレーニング 勾配 のヒストグラムを ログ します |
|  `training_data` |  (tuple) `model.fit` に渡されるのと同じ形式 `(X,y)`。これは 勾配 を計算するために必要です。`log_gradients` が `True` の場合は必須です。 |
|  `validation_data` |  (tuple) `model.fit` に渡されるのと同じ形式 `(X,y)`。wandb が 可視化 するための データの セット。これが設定されている場合、すべての エポック で、wandb は少数の 予測 を行い、後で 可視化 するために 結果 を保存します。画像データを扱っている場合は、正しく ログ するために `input_type` と `output_type` も設定してください。 |
|  `generator` |  (generator) wandb が 可視化 するための 検証データ を返す generator。この generator は、タプル `(X,y)` を返す必要があります。wandb が特定の データ 例を 可視化 するには、`validate_data` または generator のいずれかを設定する必要があります。画像データを扱っている場合は、正しく ログ するために `input_type` と `output_type` も設定してください。 |
|  `validation_steps` |  (int) `validation_data` が generator の場合、完全な 検証セット に対して generator を実行するステップ数。 |
|  `labels` |  (list) wandb で データを 可視化 している場合、この ラベル のリストは、多クラス分類器を構築している場合に数値出力を理解可能な文字列に変換します。バイナリ分類器を作成している場合は、2 つの ラベル のリスト ["false の ラベル ", "true の ラベル "] を渡すことができます。`validate_data` と generator が両方とも false の場合、これは何も行いません。 |
|  `predictions` |  (int) 各 エポック で 可視化 するために行う 予測 の数。最大は 100 です。 |
|  `input_type` |  (string) 可視化 を支援するための model 入力のタイプ。次のいずれかになります: (`image`, `images`, `segmentation_mask`, `auto`)。 |
|  `output_type` |  (string) 可視化 を支援するための model 出力のタイプ。次のいずれかになります: (`image`, `images`, `segmentation_mask`, `label`)。 |
|  `log_evaluation` |  (boolean) True の場合、各 エポック で 検証データ と model の 予測 を含む Table を保存します。詳細については、`validation_indexes`、`validation_row_processor`、および `output_row_processor` を参照してください。 |
|  `class_colors` |  ([float, float, float]) 入力または出力が セグメンテーションマスク の場合、各クラスの rgb タプル (範囲 0 ～ 1) を含む配列。 |
|  `log_batch_frequency` |  (integer) None の場合、 コールバック はすべての エポック を ログ します。整数に設定すると、 コールバック は `log_batch_frequency` バッチごとに トレーニング メトリクス を ログ します。 |
|  `log_best_prefix` |  (string) None の場合、追加の概要 メトリクス は保存されません。文字列に設定すると、監視対象の メトリクス と エポック にこの 値 が付加され、概要 メトリクス として保存されます。 |
|  `validation_indexes` |  ([wandb.data_types._TableLinkMixin]) 各 検証 例に関連付ける インデックス キー の順序付きリスト。log_evaluation が True で、`validation_indexes` が指定されている場合、 検証データ の Table は作成されず、代わりに各 予測 が `TableLinkMixin` で表される行に関連付けられます。このような キー を取得する最も一般的な方法は、行 キー のリストを返す `Table.get_index()` を使用することです。 |
|  `validation_row_processor` |  (Callable) 検証データ に適用する関数。通常は データを 可視化 するために使用されます。この関数は、`ndx` (int) と `row` (dict) を受け取ります。model に単一の入力がある場合、`row["input"]` はその行の入力 データ になります。それ以外の場合は、入力 スロット の名前に基づいて キー が設定されます。fit 関数が単一のターゲットを受け取る場合、`row["target"]` はその行のターゲット データ になります。それ以外の場合は、出力 スロット の名前に基づいて キー が設定されます。たとえば、入力 データ が単一の ndarray であるが、データを Image として 可視化 したい場合は、`lambda ndx, row: {"img": wandb.Image(row["input"])}` を プロセッサ として指定できます。log_evaluation が False の場合、または `validation_indexes` が存在する場合は無視されます。 |
|  `output_row_processor` |  (Callable) `validation_row_processor` と同じですが、model の出力に適用されます。`row["output"]` には、model 出力の 結果 が含まれます。 |
|  `infer_missing_processors` |  (bool) `validation_row_processor` と `output_row_processor` が見つからない場合に推論する必要があるかどうかを決定します。デフォルトは True です。`labels` が指定されている場合は、必要に応じて分類タイプの プロセッサ を推論しようとします。 |
|  `log_evaluation_frequency` |  (int) 評価 結果 を ログ する頻度を決定します。デフォルトは 0 (トレーニング の最後にのみ) です。すべての エポック で ログ する場合は 1 に、他のすべての エポック で ログ する場合は 2 に設定します。log_evaluation が False の場合は効果がありません。 |
|  `compute_flops` |  (bool) Keras Sequential または Functional model の FLOP を GigaFLOP 単位で計算します。 |

## メソッド

### `get_flops`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L1045-L1091)

```python
get_flops() -> float
```

推論モードで tf.keras.Model または tf.keras.Sequential model の FLOPS [GFLOPs] を計算します。

内部的には tf.compat.v1.profiler を使用します。

### `set_model`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L567-L576)

```python
set_model(
    model
)
```

### `set_params`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L564-L565)

```python
set_params(
    params
)
```