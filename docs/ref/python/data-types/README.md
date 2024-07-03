# データタイプ

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

このモジュールは、W&B にリッチでインタラクティブな可視化をログするためのデータタイプを定義します。

データタイプには、画像、オーディオ、ビデオなどの一般的なメディアタイプ、テーブルやHTMLなどの情報の柔軟なコンテナなどが含まれます。

メディアのログに関しては、[こちらのガイド](https://docs.wandb.com/guides/track/log/media)を参照してください。

インタラクティブなデータセットとモデル分析のための構造化データのログに関しては、[W&B Tables ガイド](https://docs.wandb.com/guides/data-vis)を参照してください。

これらの特別なデータタイプはすべて WBValue のサブクラスです。すべてのデータタイプはJSONにシリアライズされます。これは、wandb がオブジェクトをローカルに保存し、W&B サーバーにアップロードするために使用する形式です。

## クラス

[`class Audio`](./audio.md): オーディオクリップのための Wandb クラス。

[`class BoundingBoxes2D`](./boundingboxes2d.md): 2Dバウンディングボックスオーバーレイで画像をフォーマットし、W&B にログするための形式。

[`class Graph`](./graph.md): グラフのための Wandb クラス。

[`class Histogram`](./histogram.md): ヒストグラムのための wandb クラス。

[`class Html`](./html.md): 任意の html のための Wandb クラス。

[`class Image`](./image.md): 画像を W&B にログするための形式。

[`class ImageMask`](./imagemask.md): 画像マスクやオーバーレイを W&B にログするための形式。

[`class Molecule`](./molecule.md): 3D分子データのための Wandb クラス。

[`class Object3D`](./object3d.md): 3Dポイントクラウドのための Wandb クラス。

[`class Plotly`](./plotly.md): plotly プロットのための Wandb クラス。

[`class Table`](./table.md): 表形式データを表示および分析するための Table クラス。

[`class Video`](./video.md): ビデオを W&B にログするための形式。

[`class WBTraceTree`](./wbtracetree.md): トレースツリーデータのためのメディアオブジェクト。