
# Python ライブラリ

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

wandb を使用して機械学習の作業を追跡します。

最も一般的に使用される関数/オブジェクトは次のとおりです:

- wandb.init — トレーニングスクリプトの最上部で新しい run を初期化
- wandb.config — ハイパーパラメーターとメタデータを追跡
- wandb.log — トレーニングループ内で時間経過とともにメトリクスとメディアをログ

ガイドと例については、https://docs.wandb.ai を参照してください。

スクリプトとインタラクティブノートブックについては、https://github.com/wandb/examples を参照してください。

リファレンスドキュメントについては、https://docs.wandb.com/ref/python を参照してください。

## クラス

[`class Artifact`](./artifact.md): データセットとモデルのバージョン管理のための柔軟で軽量な構成要素。

[`class Run`](./run.md): wandb によってログされる計算の単位。通常、これは ML 実験です。

## 関数

[`agent(...)`](./agent.md): 1つ以上の sweep agents を開始します。

[`controller(...)`](./controller.md): パブリックな sweep コントローラのコンストラクタ。

[`finish(...)`](./finish.md): run を終了としてマークし、すべてのデータのアップロードを完了します。

[`init(...)`](./init.md): 新しい run を開始して W&B にトラッキングとログを行います。

[`log(...)`](./log.md): 現在の run の履歴に辞書形式のデータをログします。

[`login(...)`](./login.md): W&B ログイン資格情報を設定します。

[`save(...)`](./save.md): 1つ以上のファイルを W&B に同期します。

[`sweep(...)`](./sweep.md): ハイパーパラメーター探索を初期化します。

[`watch(...)`](./watch.md): torch モデルにフックして勾配とトポロジーを収集します。

| その他のメンバー |  |
| :--- | :--- |
|  `__version__`<a id="__version__"></a> |  `'0.17.1'` |
|  `config`<a id="config"></a> |   |
|  `summary`<a id="summary"></a> |   |