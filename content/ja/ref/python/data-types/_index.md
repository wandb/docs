---
title: Data Types
menu:
  reference:
    identifier: ja-ref-python-data-types-_index
---

このモジュールでは、リッチでインタラクティブな 可視化 を W&B に ログ するための データ 型を定義します。

データ 型には、画像、音声、ビデオなどの一般的なメディア タイプ、テーブル や HTML などの情報の柔軟なコンテナなどが含まれます。

メディアの ログ に関する詳細については、[ガイド](https://docs.wandb.com/guides/track/log/media) を参照してください。

インタラクティブな データセット および model の 分析 用に構造化された データを ログ する方法の詳細については、[W&B Tables に関するガイド](https://docs.wandb.com/guides/models/tables/) を参照してください。

これらの特殊な データ 型はすべて、WBValue のサブクラスです。すべての データ 型は JSON にシリアライズされます。これは、wandb が オブジェクト をローカルに保存し、W&B サーバー にアップロードするために使用する方法だからです。

## クラス

[`class Audio`](./audio.md): オーディオクリップ用の Wandb クラス。

[`class BoundingBoxes2D`](./boundingboxes2d.md): W&B に ログ するための2D バウンディングボックス オーバーレイを含む画像形式。

[`class Graph`](./graph.md): グラフ用の Wandb クラス。

[`class Histogram`](./histogram.md): ヒストグラム用の wandb クラス。

[`class Html`](./html.md): 任意の html 用の Wandb クラス。

[`class Image`](./image.md): W&B に ログ するための画像形式。

[`class ImageMask`](./imagemask.md): W&B に ログ するための画像マスクまたはオーバーレイ形式。

[`class Molecule`](./molecule.md): 3D 分子 データ 用の Wandb クラス。

[`class Object3D`](./object3d.md): 3D ポイントクラウド用の Wandb クラス。

[`class Plotly`](./plotly.md): plotly プロット用の Wandb クラス。

[`class Table`](./table.md): 表形式の データを表示および 分析 するために使用される Table クラス。

[`class Video`](./video.md): W&B に ログ するためのビデオ形式。

[`class WBTraceTree`](./wbtracetree.md): トレース ツリー データ 用のメディア オブジェクト。
