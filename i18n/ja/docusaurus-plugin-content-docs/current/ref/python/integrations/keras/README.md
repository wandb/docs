# Keras

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/__init__.py)

`wandb`と[`Keras`](https://keras.io/)を連携させるためのツール。

Kerasは、[`TensorFlow`](https://www.tensorflow.org/)のディープラーニングAPIです。

## クラス

[`class WandbCallback`](./wandbcallback.md): `WandbCallback`は、Kerasとwandbを自動的に連携させます。

[`class WandbEvalCallback`](./wandbevalcallback.md): モデル予測の可視化のためのKerasコールバックを構築するための抽象基本クラス。

[`class WandbMetricsLogger`](./wandbmetricslogger.md): システムメトリクスをW&Bに送信するロガー。

[`class WandbModelCheckpoint`](./wandbmodelcheckpoint.md): 定期的にKerasモデルやモデルの重みを保存するチェックポイント。