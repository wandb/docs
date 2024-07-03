# Python Library

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandb を使用して機械学習作業をトラックします。

最もよく使用される関数/オブジェクトは次の通りです:

- wandb.init — トレーニングスクリプトの先頭で新しい run を初期化
- wandb.config — ハイパーパラメーターとメタデータをトラック
- wandb.log — トレーニングループ内で時間と共にメトリクスとメディアをログ

ガイドと例については、https://docs.wandb.ai を参照してください。

スクリプトとインタラクティブなノートブックについては、https://github.com/wandb/examples を参照してください。

リファレンスドキュメントについては、https://docs.wandb.com/ref/python を参照してください。

## Classes

[`class Artifact`](./artifact.md): データセットとモデルのバージョン管理のための柔軟で軽量なビルディングブロック。

[`class Run`](./run.md): wandb によってログされる計算単位。通常、これは ML 実験です。

## Functions

[`agent(...)`](./agent.md): 一つ以上の sweep agent を開始。

[`controller(...)`](./controller.md): 公開された sweep controller のコンストラクタ。

[`finish(...)`](./finish.md): run を完了としてマークし、すべてのデータのアップロードを完了。

[`init(...)`](./init.md): 新しい run を開始して W&B にトラックおよびログ。

[`log(...)`](./log.md): 現在の run の履歴にデータの辞書をログ。

[`login(...)`](./login.md): W&B ログイン資格情報を設定。

[`save(...)`](./save.md): 一つ以上のファイルを W&B に同期。

[`sweep(...)`](./sweep.md): ハイパーパラメーター探索を初期化。

[`watch(...)`](./watch.md): 勾配とトポロジーを収集するために torch モデルにフック。

| Other Members |  |
| :--- | :--- |
|  `__version__`<a id="__version__"></a> |  `'0.17.3'` |
|  `config`<a id="config"></a> |   |
|  `summary`<a id="summary"></a> |   |