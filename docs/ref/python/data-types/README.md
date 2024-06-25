
# Data Types

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

このモジュールは、W&Bにリッチでインタラクティブな可視化をログするためのデータタイプを定義しています。

データタイプには、画像、音声、ビデオのような一般的なメディアタイプ、テーブルやHTMLのような情報の柔軟なコンテナなどがあります。

メディアのログに関する詳細は、[ガイド](https://docs.wandb.com/guides/track/log/media)を参照してください。

インタラクティブなデータセットやモデル分析のための構造化データのログに関する詳細は、[W&B Tablesのガイド](https://docs.wandb.com/guides/data-vis)を参照してください。

これらの特別なデータタイプはすべてWBValueのサブクラスです。すべてのデータタイプはJSONにシリアライズされます。これは、wandbがオブジェクトをローカルに保存し、W&Bサーバーにアップロードするために使用する形式だからです。

## クラス

[`class Audio`](./audio.md): 音声クリップのためのWandbクラス。

[`class BoundingBoxes2D`](./boundingboxes2d.md): 2D境界ボックスオーバーレイを持つ画像をフォーマットしてW&Bにログします。

[`class Graph`](./graph.md): グラフのためのWandbクラス。

[`class Histogram`](./histogram.md): ヒストグラムのためのwandbクラス。

[`class Html`](./html.md): 任意のHTMLのためのWandbクラス。

[`class Image`](./image.md): 画像をフォーマットしてW&Bにログします。

[`class ImageMask`](./imagemask.md): 画像マスクまたはオーバーレイをフォーマットしてW&Bにログします。

[`class Molecule`](./molecule.md): 3D分子データのためのWandbクラス。

[`class Object3D`](./object3d.md): 3DポイントクラウドのためのWandbクラス。

[`class Plotly`](./plotly.md): PlotlyプロットのためのWandbクラス。

[`class Table`](./table.md): 表データを表示および分析するためのTableクラス。

[`class Video`](./video.md): ビデオをフォーマットしてW&Bにログします。

[`class WBTraceTree`](./wbtracetree.md): トレースツリーデータのためのメディアオブジェクト。