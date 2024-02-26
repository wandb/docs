# インポート & エクスポート API
[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/__init__.py)
Public APIを使って、W&Bに保存したデータをエクスポートしたり更新したりします。
このAPIを使う前に、スクリプトからデータをログに記録しておく必要があります。詳細は[クイックスタート](https://docs.wandb.ai/quickstart)を参照してください。
Public APIを使って、
 - Jupyterノートブックで事後分析を行うために、結果をデータフレームとしてダウンロードする
などができます。
Public APIの使用方法については、[ガイド](https://docs.wandb.com/guides/track/public-api-guide)をチェックしてください。
## クラス
[`class Api`](./api.md): wandbサーバーへの問い合わせに使用されます。
`class Artifact`: wandbのアーティファクト。
[`class File`](./file.md): wandbによって保存されたファイルに関連するクラス。
[`class Files`](./files.md): `File`オブジェクトの反復可能なコレクション。








