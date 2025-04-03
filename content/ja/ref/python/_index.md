---
title: Python Library
menu:
  reference:
    identifier: ja-ref-python-_index
---

wandb を使用して、機械学習の作業を追跡します。

モデルの学習とファインチューン、実験からプロダクションまでのモデルを管理します。

ガイド と例については、https://docs.wandb.ai を参照してください。

スクリプト とインタラクティブな notebook については、https://github.com/wandb/examples を参照してください。

リファレンスドキュメントについては、https://docs.wandb.com/ref/python を参照してください。

## クラス

[`class Artifact`](./artifact.md): データセット とモデルの バージョン管理のための、柔軟で軽量な構成要素。

[`class Run`](./run.md): wandb によって記録される計算の単位。通常、これは ML の実験です。

## 関数

[`agent(...)`](./agent.md): 1つまたは複数の sweep agent を起動します。

[`controller(...)`](./controller.md): パブリック sweep controller コンストラクタ。

[`finish(...)`](./finish.md): run を終了し、残りのデータをアップロードします。

[`init(...)`](./init.md): 新しい run を開始して、W&B への追跡とログ記録を行います。

[`log(...)`](./log.md): run データをアップロードします。

[`login(...)`](./login.md): W&B のログイン認証情報を設定します。

[`save(...)`](./save.md): 1つまたは複数のファイルを W&B に同期します。

[`sweep(...)`](./sweep.md): ハイパーパラメーター探索 を初期化します。

[`watch(...)`](./watch.md): 指定された PyTorch モデルにフックして、勾配 とモデルの計算グラフを監視します。

| その他のメンバー |  |
| :--- | :--- |
| `__version__`<a id="__version__"></a> | `'0.19.8'` |
| `config`<a id="config"></a> |   |
| `summary`<a id="summary"></a> |   |
