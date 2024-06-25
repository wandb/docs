
# WandbModelCheckpoint

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/integration/keras/callbacks/model_checkpoint.py#L27-L200' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Kerasモデルやモデルの重みを定期的に保存するチェックポイントです。

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

保存された重みは`wandb.Artifact`としてW&Bにアップロードされます。

このコールバックは`tf.keras.callbacks.ModelCheckpoint`をサブクラス化しているため、チェックポイントのロジックは親コールバックにより処理されます。詳しくはこちらをご覧ください: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

このコールバックは、`model.fit()`を使用してトレーニングする際に、モデルや重みをあるインターバルでチェックポイントファイルに保存するために使用されます。モデルチェックポイントはW&B Artifactsとしてログされます。詳しくはこちらをご覧ください:
https://docs.wandb.ai/guides/artifacts

このコールバックは以下の機能を提供します:
- "monitor"に基づいて「最高のパフォーマンス」を達成したモデルを保存します。
- パフォーマンスに関係なく、各エポックの終わりにモデルを保存します。
- エポックの終わりや一定数のトレーニングバッチ後にモデルを保存します。
- モデルの重みのみを保存するか、モデル全体を保存します。
- モデルをSavedModel形式または`.h5`形式で保存します。

| 引数 |  |
| :--- | :--- |
| `filepath` |  (Union[str, os.PathLike]) モデルファイルを保存するパス。`filepath`には名前付きフォーマットオプションを含めることができ、エポック番号や`logs`（`on_epoch_end`で渡される）のキーの値によって埋められます。例えば、`filepath`が`model-{epoch:02d}-{val_loss:.2f}`の場合、モデルチェックポイントはエポック番号と検証損失がファイル名に含まれて保存されます。 |
| `monitor` |  (str) 監視するメトリクスの名前。デフォルトは"val_loss"です。 |
| `verbose` |  (int) 冗長モード、0または1。モード0は無音、モード1はコールバックがアクションを取るときにメッセージを表示します。 |
| `save_best_only` |  (bool) `save_best_only=True`の場合、モデルが「最高」とみなされるときにのみ保存され、最新の最高モデルは監視量に従って上書きされません。`filepath`に`{epoch}`などのフォーマットオプションが含まれていない場合、`filepath`は新しい優れたモデルによってローカルに上書きされます。アーティファクトとしてログされたモデルは引き続き正しい`monitor`と関連付けられます。Artifactsは継続的にアップロードされ、新しい最高モデルが見つかるたびに別々にバージョン管理されます。 |
| `save_weights_only` |  (bool) Trueの場合、モデルの重みのみが保存されます。 |
| `mode` |  (Mode) {'auto', 'min', 'max'}のいずれか。`val_acc`の場合は`max`、`val_loss`の場合は`min`が適切です。 |
| `save_freq` |  (Union[SaveStrategy, int]) `epoch`または整数。`'epoch'`を使用すると、各エポックの後にモデルが保存されます。整数を使用すると、このバッチ数の終わりにモデルが保存されます。`val_acc`や`val_loss`などの検証メトリクスを監視する際は、これらのメトリクスがエポックの終わりにのみ利用可能であるため、save_freqは"epoch"に設定する必要があります。 |
| `options` |  (Optional[str]) `save_weights_only`がTrueの場合はオプションの`tf.train.CheckpointOptions`オブジェクト、`save_weights_only`がFalseの場合はオプションの`tf.saved_model.SaveOptions`オブジェクト。 |
| `initial_value_threshold` |  (Optional[float]) 監視するメトリクスの最初の「最高」値。 |

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