---
title: CSV メトリクスのエクスポートでステップが抜けているのはなぜですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
- run
---

エクスポート制限により、run 全体の履歴を CSV 形式でエクスポートしたり、`run.history` API を使用して取得することができない場合があります。run の完全な履歴にアクセスするには、Parquet 形式で run history artifact をダウンロードしてください。

```python
import wandb
import pandas as pd

run = wandb.init()
# run history artifact を取得します
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
# Parquet ファイルを読み込んでデータフレームとして扱います
df = pd.read_parquet('<path to .parquet file>')
```