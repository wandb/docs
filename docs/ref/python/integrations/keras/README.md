# Keras

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/integration/keras/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

`wandb` を [`Keras`](https://keras.io/) と統合するためのツールです。

Kerasは、[`TensorFlow`](https://www.tensorflow.org/) のためのディープラーニングAPIです。

## クラス

[`class WandbCallback`](./wandbcallback.md): `WandbCallback` は Keras と wandb を自動で統合します。

[`class WandbEvalCallback`](./wandbevalcallback.md): モデル予測の可視化のための Keras コールバックを構築するための抽象基底クラスです。

[`class WandbMetricsLogger`](./wandbmetricslogger.md): システムメトリクスを W&B に送信するロガーです。

[`class WandbModelCheckpoint`](./wandbmodelcheckpoint.md): Kerasモデルまたはモデルの重みを定期的に保存するためのチェックポイントです。