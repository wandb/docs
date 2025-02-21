---
title: Prodigy
description: W&B を Prodigy と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) は、機械学習モデルのトレーニングと評価データを作成するためのアノテーションツールであり、エラー分析、データ検査、データクリーニングにも利用できます。[W&B Tables]({{< relref path="/guides/core/tables/tables-walkthrough.md" lang="ja" >}}) は、データセット（およびその他の項目）を W&B 内でログ、可視化、分析、共有することができます。

[Prodigy との W&B インテグレーション](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) により、Prodigy でアノテーションされたデータセットを W&B に直接アップロードし、Tables と一緒に使用するための簡単で使いやすい機能が追加されます。

次のようなコードを数行実行します：

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

すると、このような視覚的でインタラクティブな共有可能なテーブルが得られます：

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="" >}}

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使用して、アノテーションされた prodigy データセットをローカルの Prodigy データベースから W&B に直接 [Table]({{< relref path="/ref/python/data-types/table" lang="ja" >}}) フォーマットでアップロードします。Prodigy のインストールとセットアップを含む詳細情報については、[Prodigy のドキュメント](https://prodi.gy/docs/) を参照してください。

W&B は、自動的に画像と命名されたエンティティフィールドを [`wandb.Image`]({{< relref path="/ref/python/data-types/image" lang="ja" >}}) と [`wandb.Html`]({{< relref path="/ref/python/data-types/html" lang="ja" >}}) にそれぞれ変換しようとします。これらの可視化を含めるために、結果のテーブルに追加の列が追加される場合があります。

## 詳細な例を読みましょう

W&B Prodigy インテグレーションで生成された例の可視化については、[W&B Tables を使用した Prodigy Datasets の視覚化](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) をご覧ください。  

## spaCy も使用していますか？

W&B は spaCy とのインテグレーションも提供しており、[こちらのドキュメント]({{< relref path="/guides/integrations/spacy" lang="ja" >}})を参照してください。