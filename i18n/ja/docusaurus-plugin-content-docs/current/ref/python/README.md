# Pythonライブラリ


[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c505c66a5f9c1530671564dae3e9e230f72f6584/wandb/__init__.py)



wandbを使って機械学習作業を追跡します。


最もよく使用される関数/オブジェクトは以下の通りです：
 - wandb.init — トレーニングスクリプトの最上部で新しいrunを初期化します
 - wandb.config — ハイパーパラメーターとメタデータを追跡します
 - wandb.log — トレーニングループ内でメトリクスとメディアを経時的に記録します

ガイドと例については、https://docs.wandb.aiを参照してください。

スクリプトとインタラクティブなノートブックについては https://github.com/wandb/examplesを参照してください。

参考文献については、https://docs.wandb.com/ref/pythonを参照してください。

## クラス​

[`class Artifact`](./artifact.md): データセットとモデルバージョン管理用の、柔軟かつ軽量なビルディングブロック。

[`class Run`](./run.md): wandbによって記録される計算の単位。通常これはML実験になります。

## 関数​

[`agent(...)`](./agent.md): CLIまたはjupyterによって使用される、包括的なエージェントエントリポイント。

[`controller(...)`](./controller.md): パブリックスウィープコントローラコンストラクタ。

[`finish(...)`](./finish.md): runに`終了`のマークを付け、全データのアップロードを終了します。

[`init(...)`](./init.md): 新しいrunを開始し、追跡してW&Bに記録します。

[`log(...)`](./log.md): データの辞書を現在のrunの履歴に追加します。

[`save(...)`](./save.md): 指定されたポリシーで、glob_strと一致するすべてのファイルをwandbと同期化します。

[`sweep(...)`](./sweep.md): ハイパーパラメータースウィープを初期化します。

[`watch(...)`](./watch.md): torchモデルを接続し、勾配とトポロジを収集します。




| その他のメンバー | |
| :--- | :--- |
| `__version__` | `'0.13.11'` |
| `config` | |
| `summary` | |

