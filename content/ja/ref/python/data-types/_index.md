---
title: データタイプ
menu:
  reference:
    identifier: ja-ref-python-data-types-_index
---

このモジュールは、W&B にリッチでインタラクティブな可視化をログするためのデータ型を定義します。

データ型には、画像、オーディオ、ビデオなどの一般的なメディアタイプや、テーブルや HTML などの情報を柔軟に格納するコンテナが含まれます。

メディアのログの詳細については、[ガイド](https://docs.wandb.com/guides/track/log/media)をご覧ください。

インタラクティブな データセット と モデル分析 のための構造化データのログの詳細については、[W&B Tables のガイド](https://docs.wandb.com/guides/models/tables/)をご覧ください。

これらの特別なデータ型はすべて WBValue のサブクラスです。すべてのデータ型は JSON にシリアライズされます。wandb はこれを使用して オブジェクト をローカルに保存し、W&B サーバー にアップロードします。

## クラス

[`class Audio`](./audio.md): オーディオクリップ用の Wandb クラス。

[`class BoundingBoxes2D`](./boundingboxes2d.md): 画像を 2D バウンディングボックスオーバーレイでフォーマットし、W&Bにログします。

[`class Graph`](./graph.md): グラフ用の Wandb クラス。

[`class Histogram`](./histogram.md): ヒストグラム用の wandb クラス。

[`class Html`](./html.md): 任意の html 用の Wandb クラス。

[`class Image`](./image.md): 画像をフォーマットして W&Bにログします。

[`class ImageMask`](./imagemask.md): 画像マスクやオーバーレイをフォーマットし、W&Bにログします。

[`class Molecule`](./molecule.md): 3D 分子データ用の Wandb クラス。

[`class Object3D`](./object3d.md): 3D ポイントクラウド用の Wandb クラス。

[`class Plotly`](./plotly.md): plotly プロット用の Wandb クラス。

[`class Table`](./table.md): 表形式のデータを表示および分析するための Table クラス。

[`class Video`](./video.md): ビデオをフォーマットして W&Bにログします。

[`class WBTraceTree`](./wbtracetree.md): トレースツリーデータのためのメディアオブジェクト。