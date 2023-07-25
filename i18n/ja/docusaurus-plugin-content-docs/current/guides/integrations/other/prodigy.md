---
slug: /guides/integrations/prodigy
description: How to integrate W&B with Prodigy.
displayed_sidebar: ja
---

# Prodigy

[Prodigy](https://prodi.gy/)は、機械学習モデルのトレーニングや評価データ、エラー分析、データ検査・クリーニングのための注釈ツールです。[W&Bテーブル](../../tables/tables-quickstart.md)を使用すると、データセット（やそれ以上のもの！）をW&B内にログ、可視化、分析、共有することができます。

[ProdigyとW&Bの統合](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py)により、Prodigyで注釈付きデータセットをW&Bに直接アップロードし、テーブルで使用するためのシンプルで使いやすい機能が追加されます。

以下のようなコードを数行実行してください。

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

すると、このような視覚的でインタラクティブな、共有可能なテーブルが得られます。

![](/images/integrations/prodigy_interactive_visual.png)

## クイックスタート

`wandb.integration.prodigy.upload_dataset`を使用して、ローカルのProdigyデータベースから注釈付きのProdigyデータセットをW&Bの[Table](https://docs.wandb.ai/ref/python/data-types/table)形式で直接アップロードします。Prodigyの詳細情報、インストール・セットアップについては、[Prodigyドキュメント](https://prodi.gy/docs/)を参照してください。

W&Bは、画像や名前付きエンティティフィールドを自動的に[`wandb.Image`](https://docs.wandb.ai/ref/python/data-types/image) や [`wandb.Html`](https://docs.wandb.ai/ref/python/data-types/html) に変換しようとします。結果の表には、これらの可視化を含めるために追加の列が追加されることがあります。
## 詳細な例を読む

W&B Prodigy統合で生成された例の可視化については、[Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc)をご覧ください。

## spaCyも使っていますか？

W&BにはspaCyとの統合もあります。[こちらのドキュメント](https://docs.wandb.ai/guides/integrations/spacy)をご覧ください。