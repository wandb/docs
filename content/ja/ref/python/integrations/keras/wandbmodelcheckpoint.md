---
title: WandbModelCheckpoint
menu:
  reference:
    identifier: ja-ref-python-integrations-keras-wandbmodelcheckpoint
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/model_checkpoint.py#L20-L188 >}}

Keras の model または model の重みを定期的に保存するチェックポイントです。

```python
WandbModelCheckpoint(
    filepath: StrPath,
    monitor: str = "val_loss",
    verbose: int = 0,
    save_best_only: bool = (False),
    save_weights_only: bool = (False),
    mode: Mode = "auto",
    save_freq: Union[SaveStrategy, int] = "epoch",
    initial_value_threshold: Optional[float] = None,
    **kwargs
) -> None
```

保存された重みは、 `wandb.Artifact` として W&B にアップロードされます。

この callback は `tf.keras.callbacks.ModelCheckpoint` からサブクラス化されているため、
チェックポイントのロジックは親の callback によって処理されます。詳細については、
こちらをご覧ください: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

この callback は、`model.fit()` を使用した training と組み合わせて使用​​し、
model または重み (チェックポイントファイル内) を一定の間隔で保存します。 model のチェックポイントは、W&B Artifacts としてログに記録されます。詳細については、こちらをご覧ください:
https://docs.wandb.ai/guides/artifacts

この callback は、次の機能を提供します。
- 「monitor」に基づいて「最高のパフォーマンス」を達成した model を保存します。
- パフォーマンスに関係なく、エポックの最後に毎回 model を保存します。
- エポックの終わり、または固定数の training バッチの後に model を保存します。
- model の重みのみを保存するか、model 全体を保存します。
- SavedModel 形式または `.h5` 形式で model を保存します。

| Args |  |
| :--- | :--- |
|  `filepath` |  (Union[str, os.PathLike]) model ファイルを保存するパス。 `filepath` には、`epoch` の値と `logs` ( `on_epoch_end` で渡される) のキーによって入力される名前付きの書式設定オプションを含めることができます。たとえば、`filepath` が `model-{epoch:02d}-{val_loss:.2f}` の場合、model のチェックポイントは、ファイル名にエポック番号と検証損失を付けて保存されます。 |
|  `monitor` |  (str) 監視するメトリクスの名前。デフォルトは "val_loss" です。 |
|  `verbose` |  (int) 冗長モード、0 または 1。モード 0 はサイレントで、モード 1 は callback がアクションを実行するときにメッセージを表示します。 |
|  `save_best_only` |  (bool) `save_best_only=True` の場合、model が「最高」と見なされる場合にのみ保存され、監視対象の量に応じて、最新の最高の model は上書きされません。 `filepath` に `{epoch}` のような書式設定オプションが含まれていない場合、`filepath` はローカルで新しいより良い model によって上書きされます。 Artifact としてログに記録された model は、引き続き正しい `monitor` に関連付けられます。 Artifacts は継続的にアップロードされ、新しい最高の model が見つかると、個別にバージョン管理されます。 |
|  `save_weights_only` |  (bool) True の場合、model の重みのみが保存されます。 |
|  `mode` |  (Mode) {'auto', 'min', 'max'} のいずれか。 `val_acc` の場合、これは `max` である必要があり、`val_loss` の場合、これは `min` である必要があります。 |
|  `save_freq` |  (Union[SaveStrategy, int]) `epoch` または整数。 `'epoch'` を使用すると、callback は各エポックの後に model を保存します。整数を使用すると、callback はこの多数のバッチの終わりに model を保存します。 `val_acc` や `val_loss` などの検証メトリクスを監視する場合、これらのメトリクスはエポックの最後にのみ使用可能であるため、save_freq を "epoch" に設定する必要があることに注意してください。 |
|  `initial_value_threshold` |  (Optional[float]) 監視対象のメトリクスの浮動小数点初期「最良」値。 |

| Attributes |  |
| :--- | :--- |

## Methods

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