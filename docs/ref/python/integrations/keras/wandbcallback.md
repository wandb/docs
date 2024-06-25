
# WandbCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/keras.py#L291-L1080' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

`WandbCallback`はkerasとwandbを自動的にインテグレーションします。

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

`WandbCallback`はkerasが収集した任意のメトリクスからの履歴データを自動的にログします。損失や`keras_model.compile()`に渡されたものを記録します。

`WandbCallback`は、`monitor`および`mode`属性によって定義された "ベスト" トレーニングステップに関連するrunのサマリーメトリクスを設定します。これはデフォルトで最小の`val_loss`を持つエポックに設定されます。`WandbCallback`はデフォルトで最も良いエポックに関連するモデルを保存します。

`WandbCallback`は勾配とパラメータのヒストグラムをオプションでログすることができます。

`WandbCallback`はwandbが可視化するためのトレーニングと検証データをオプションで保存することができます。

| 引数 |  |
| :--- | :--- |
|  `monitor` |  (str) モニタリングするメトリクスの名前。デフォルトは`val_loss`。 |
|  `mode` |  (str) {`auto`、`min`、`max`}のいずれか。 `min` - モニターが最小化されたときにモデルを保存 `max` - モニターが最大化されたときにモデルを保存 `auto` - モデルを保存するタイミングを自動的に決定（デフォルト）。 |
|  `save_model` |  True - モニターがすべての前のエポックを上回ったときにモデルを保存 False - モデルを保存しない |
|  `save_graph` |  (boolean) Trueの場合、モデルのグラフをwandbに保存します（デフォルトはTrue）。 |
|  `save_weights_only` |  (boolean) Trueの場合、モデルの重みのみが保存されます（`model.save_weights(filepath)`）。さもなくば完全なモデルが保存されます（`model.save(filepath)`）。 |
|  `log_weights` |  (boolean) Trueの場合、モデルのレイヤーの重みのヒストグラムを保存します。 |
|  `log_gradients` |  (boolean) Trueの場合、トレーニング勾配のヒストグラムをログします。 |
|  `training_data` |  (tuple) `model.fit`に渡される形式`(X, y)`。これは勾配計算のために必要です。`log_gradients`がTrueの場合、これが必須です。 |
|  `validation_data` |  (tuple) `model.fit`に渡される形式`(X, y)`。wandbが可視化するためのデータセット。これが設定されている場合、毎エポックごとにwandbは少数の予測を行い、後で可視化するために結果を保存します。画像データを扱っている場合は、正しくログするために`input_type`と`output_type`も設定してください。 |
|  `generator` |  (generator) wandbが可視化するための検証データを返すジェネレーター。このジェネレーターはタプル`(X, y)`を返すべきです。`validate_data`またはジェネレーターのどちらかが設定されていれば、wandbは特定のデータ例を可視化します。画像データを扱っている場合は、正しくログするために`input_type`と`output_type`も設定してください。 |
|  `validation_steps` |  (int) `validation_data`がジェネレーターの場合、検証セット全体についてジェネレーターを実行するステップ数。 |
|  `labels` |  (list) データをwandbで可視化する場合、このラベルのリストは数値出力を理解可能な文字列に変換します。マルチクラス分類器を構築している場合に役立ちます。バイナリ分類器を作成している場合、2つのラベル["falseのラベル", "trueのラベル"]のリストを渡すことができます。`validate_data`とジェネレーターの両方がfalseの場合、これは何も行いません。 |
|  `predictions` |  (int) 各エポックで可視化する予測の数。最大は100。 |
|  `input_type` |  (string) 可視化を助けるためのモデルの入力タイプ。次のいずれか: (`image`、`images`、`segmentation_mask`、`auto`)。 |
|  `output_type` |  (string) 可視化を助けるためのモデルの出力タイプ。次のいずれか: (`image`、`images`、`segmentation_mask`、`label`)。 |
|  `log_evaluation` |  (boolean) Trueの場合、各エポックでモデルの予測を含む検証データのTableを保存します。詳細は`validation_indexes`、`validation_row_processor`、`output_row_processor`を参照。 |
|  `class_colors` |  ([float, float, float]) 入力または出力がセグメンテーションマスクの場合、各クラスのRGBタプル（範囲0-1）を含む配列。 |
|  `log_batch_frequency` |  (integer) Noneの場合、コールバックは各エポックごとにログします。整数に設定された場合、コールバックは`log_batch_frequency`バッチごとにトレーニングメトリクスをログします。 |
|  `log_best_prefix` |  (string) Noneの場合、追加のサマリーメトリクスは保存されません。文字列に設定された場合、監視されるメトリクスとエポックがこの値でプレフィックスされ、サマリーメトリクスとして保存されます。 |
|  `validation_indexes` |  ([wandb.data_types._TableLinkMixin]) 各検証例に関連付けるインデックスキーの順序付きリスト。`log_evaluation`がTrueの場合、かつ`validation_indexes`が提供されている場合、検証データのTableは作成されず、代わりに各予測は`TableLinkMixin`で表される行に関連付けられます。このようなキーを取得する最も一般的な方法は、`Table.get_index()`を使用することで行キーのリストを取得します。 |
|  `validation_row_processor` |  (Callable) 検証データに適用する関数。通常データの可視化に使用されます。この関数は`ndx`（int）および`row`（dict）を受け取ります。モデルが単一の入力を持つ場合、`row["input"]`はその行の入力データになります。それ以外の場合、入力スロットの名前に基づいてキーが設定されます。fit関数が単一のターゲットを取る場合、`row["target"]`はその行のターゲットデータになります。それ以外の場合、出力スロットの名前に基づいてキーが設定されます。例えば、入力データが単一のndarrayであるが、データを画像として可視化したい場合、`lambda ndx, row: {"img": wandb.Image(row["input"])}`をプロセッサーとして提供できます。`log_evaluation`がFalseまたは`validation_indexes`が存在する場合、無視されます。 |
|  `output_row_processor` |  (Callable) `validation_row_processor`と同様ですが、モデルの出力に適用されます。`row["output"]`にはモデル出力の結果が含まれます。 |
|  `infer_missing_processors` |  (bool) 欠落している場合、`validation_row_processor`および`output_row_processor`を推測するかどうかを決定します。デフォルトはTrueです。`labels`が提供されている場合、適切な場所に分類タイプのプロセッサーを推測しようとします。 |
|  `log_evaluation_frequency` |  (int) 評価結果をログする頻度を決定します。デフォルトは0（トレーニング終了時のみ）。1に設定すると各エポックごと、2に設定すると毎エポックごと、というようにログします。`log_evaluation`がFalseの場合は効果がありません。 |
|  `compute_flops` |  (bool) Keras SequentialまたはFunctionalモデルのFLOPsをGigaFLOPs単位で計算します。 |

## メソッド

### `get_flops`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/keras.py#L1034-L1080)

```python
get_flops() -> float
```

推論モードのtf.keras.Modelまたはtf.keras.SequentialモデルのFLOPS [GFLOPs]を計算します。

内部ではtf.compat.v1.profilerを使用しています。

### `set_model`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/keras.py#L554-L563)

```python
set_model(
    model
)
```

### `set_params`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/keras.py#L551-L552)

```python
set_params(
    params
)
```