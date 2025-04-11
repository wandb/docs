---
title: Python ライブラリ
menu:
  reference:
    identifier: ja-ref-python-_index
---

wandb を使用して機械学習の作業を追跡します。

モデルをトレーニングおよびファインチューンし、実験からプロダクションに至るまでモデルを管理します。

ガイドや例については、https://docs.wandb.ai をご覧ください。

スクリプトやインタラクティブなノートブックについては、https://github.com/wandb/examples をご覧ください。

リファレンスドキュメントについては、https://docs.wandb.com/ref/python をご覧ください。

## クラス

[`class Artifact`](./artifact.md): データセットおよびモデルのバージョン管理のための柔軟で軽量な構成要素。

[`class Run`](./run.md): wandb によってログされる計算の単位。通常、これは機械学習の実験です。

## 関数

[`agent(...)`](./agent.md): 一つ以上の sweep agent を開始します。

[`controller(...)`](./controller.md): パブリックな sweep コントローラのコンストラクタです。

[`finish(...)`](./finish.md): run を終了し、残りのデータをアップロードします。

[`init(...)`](./init.md): 新しい run を開始して W&B へ追跡しログします。

[`log(...)`](./log.md): run のデータをアップロードします。

[`login(...)`](./login.md): W&B ログイン資格情報を設定します。

[`save(...)`](./save.md): 一つ以上のファイルを W&B に同期します。

[`sweep(...)`](./sweep.md): ハイパーパラメーター探索を初期化します。

[`watch(...)`](./watch.md): 指定された PyTorch のモデルにフックし、勾配とモデルの計算グラフを監視します。

| その他のメンバー |  |
| :--- | :--- |
|  `__version__`<a id="__version__"></a> |  `'0.19.8'` |
|  `config`<a id="config"></a> |   |
|  `summary`<a id="summary"></a> |   |