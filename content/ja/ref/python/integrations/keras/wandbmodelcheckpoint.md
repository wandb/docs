---
title: WandbModelCheckpoint
menu:
  reference:
    identifier: ja-ref-python-integrations-keras-wandbmodelcheckpoint
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/model_checkpoint.py#L20-L188 >}}

定期的に Keras モデルまたはモデルの重みを保存するチェックポイント。

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

保存された重みは `wandb.Artifact` として W&B にアップロードされます。

このコールバックは `tf.keras.callbacks.ModelCheckpoint` からサブクラス化されているため、チェックポイントのロジックは親コールバックによって処理されます。詳細はこちらで学べます: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

このコールバックは `model.fit()` を使用してトレーニングを行い、一定の間隔でモデルや重みを（チェックポイントファイルに）保存するために使用します。モデルのチェックポイントは W&B Artifacts としてログされます。詳細はこちらで学べます:
https://docs.wandb.ai/guides/artifacts

このコールバックは次の機能を提供します:
- 「モニター」に基づいて「最良のパフォーマンス」を達成したモデルを保存します。
- パフォーマンスに関係なく、各エポックの終わりにモデルを保存します。
- エポックの終わり、または一定数のトレーニングバッチ後にモデルを保存します。
- モデルの重みのみを保存するか、全体のモデルを保存します。
- モデルを SavedModel 形式か `.h5` 形式で保存します。

| 引数 |  |
| :--- | :--- |
|  `filepath` |  (Union[str, os.PathLike]) モデルファイルを保存するパス。`filepath` には名前付きのフォーマット オプションを含めることができ、これには `epoch` の値および `logs` のキー（`on_epoch_end` で渡される）が埋め込まれます。たとえば、`filepath` が `model-{epoch:02d}-{val_loss:.2f}` の場合、モデルのチェックポイントはエポック番号と検証損失とともにファイル名で保存されます。 |
|  `monitor` |  (str) 監視するメトリクスの名前。デフォルトは "val_loss"。 |
|  `verbose` |  (int) 冗長モード、0 または 1。モード 0 は静かで、モード 1 はコールバックがアクションを取るときにメッセージを表示します。 |
|  `save_best_only` |  (bool) `save_best_only=True` の場合、モデルが「最良」と見なされたときのみ保存され、監視される量に基づいて最新の最良モデルは上書きされません。`filepath` に `{epoch}` などのフォーマット オプションが含まれていない場合、`filepath` はローカルで新しいより良いモデルによって上書きされます。アーティファクトとしてログされたモデルは、依然として正しい `monitor` と関連付けられます。アーティファクトは継続的にアップロードされ、新しい最良のモデルが見つかると個別にバージョン管理されます。 |
|  `save_weights_only` |  (bool) True の場合、モデルの重みのみが保存されます。 |
|  `mode` |  (Mode) {'auto', 'min', 'max'} のいずれか。`val_acc` に対しては `max`、`val_loss` に対しては `min` など。 |
|  `save_freq` |  (Union[SaveStrategy, int]) `epoch` または整数。`'epoch'` を使用する場合、コールバックは各エポックの後にモデルを保存します。整数を使用する場合、コールバックはこのバッチ数の終わりにモデルを保存します。`val_acc` や `val_loss` などの検証メトリクスを監視する場合、save_freq は「epoch」に設定する必要があります。これらのメトリクスはエポックの終わりにのみ利用可能だからです。 |
|  `initial_value_threshold` |  (Optional[float]) 監視されるメトリクスの浮動小数点数の初期「最良」値。 |

| 属性 |  |
| :--- | :--- |

## メソッド

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