# WandbModelCheckpoint

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/callbacks/model_checkpoint.py#L27-L200' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Kerasのモデルやモデルウェイトを定期的に保存するチェックポイント。

```python
WandbModelCheckpoint(
    filepath: StrPath,
    monitor: str = "val_loss",
    verbose: int = 0,
    save_best_only: bool = (False),
    save_weights_only: bool = (False),
    mode: Mode = "auto",
    save_freq: Union[SaveStrategy, int] = "epoch",
    options: Optional[str] = None,
    initial_value_threshold: Optional[float] = None,
    **kwargs
) -> None
```

保存されたウェイトはW&Bには `wandb.Artifact` としてアップロードされます。

このコールバックは `tf.keras.callbacks.ModelCheckpoint` からサブクラス化されているため、チェックポイントロジックは親コールバックによって処理されます。詳細は以下をご覧ください:
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

このコールバックは `model.fit()` を使用したトレーニングと組み合わせて使用し、一定の間隔でモデルやウェイトを（チェックポイントファイルに）保存するためのものです。モデルのチェックポイントはW&B Artifactsとしてログされます。詳細はこちら:
https://docs.wandb.ai/guides/artifacts

このコールバックには次のような機能があります:
- "monitor" に基づいて「最高のパフォーマンス」を達成したモデルを保存する
- パフォーマンスに関係なく、エポックが終わるごとにモデルを保存する
- エポックの終了時や一定のトレーニングバッチ数ごとにモデルを保存する
- モデルのウェイトのみを保存する、またはモデル全体を保存する
- SavedModelフォーマットまたは`.h5`フォーマットでモデルを保存する

| 引数  |  |
| :--- | :--- |
|  `filepath` |  (Union[str, os.PathLike]) モデルファイルを保存するパス。`filepath`には名前付きのフォーマットオプションを含めることができ、これは `epoch` の値や `logs` のキー（`on_epoch_end`で渡される）によって埋められます。例えば: `filepath` が `model-{epoch:02d}-{val_loss:.2f}` の場合、モデルのチェックポイントはエポック番号と検証損失をファイル名に含めて保存されます。 |
|  `monitor` |  (str) モニタリングするメトリクス名。デフォルトは "val_loss"。 |
|  `verbose` |  (int) 冗長モード、0または1。モード0は無言、モード1はコールバックがアクションを取るたびにメッセージを表示します。 |
|  `save_best_only` |  (bool) `save_best_only=True` の場合、モデルが「最高」と考えられるときのみ保存され、監視対象の量に基づいて最新の最良モデルは上書きされません。`filepath` に `{epoch}` のようなフォーマットオプションが含まれていない場合、`filepath` はローカルで新しい最良モデルごとに上書きされます。アーティファクトとしてログされたモデルは依然として正しい `monitor` に関連付けられます。Artifactsは新しい最良モデルが見つかるたびに継続的にアップロードされ、バージョン管理されます。 |
|  `save_weights_only` |  (bool) Trueの場合、モデルのウェイトのみが保存されます。 |
|  `mode` |  (Mode) {'auto', 'min', 'max'} のいずれか。 `val_acc` の場合、これは `max` であり、`val_loss` の場合は `min` です。 |
|  `save_freq` |  (Union[SaveStrategy, int]) `epoch` または整数。 `'epoch'` を使用する場合、コールバックは各エポック後にモデルを保存します。整数を使用する場合、コールバックはこのバッチ数の終了時にモデルを保存します。`val_acc` や `val_loss` など検証メトリクスを監視する場合、save_freq はエポックの終了時にのみこれらのメトリクスが利用可能であるため "epoch" に設定する必要があります。 |
|  `options` |  (Optional[str]) `save_weights_only` が true の場合はオプションの `tf.train.CheckpointOptions` オブジェクト、`save_weights_only` が false の場合はオプションの `tf.saved_model.SaveOptions` オブジェクト。 |
|  `initial_value_threshold` |  (Optional[float]) モニタリングされるメトリクスの初期「最高」値。 |

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