---
title: Prodigy
description: Prodigy と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) は、機械学習 モデルのトレーニング および評価データ、エラー 分析、データ検査とクリーニングを作成するためのアノテーション ツールです。[W&B Tables]({{< relref path="/guides/core/tables/tables-walkthrough.md" lang="ja" >}}) を使用すると、W&B 内でデータセット (など) をログ記録、可視化、分析、共有できます。

[Prodigy との W&B integration](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) は、Prodigy でアノテーションされたデータセットを直接 W&B にアップロードして Tables で使用するための、シンプルで使いやすい機能を追加します。

次のようなコードを数行実行します。

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

次のような、視覚的でインタラクティブな共有可能なテーブルを取得します。

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="" >}}

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使用して、アノテーションが付けられた Prodigy データセットをローカルの Prodigy データベースから W&B に [Table]({{< relref path="/ref/python/data-types/table" lang="ja" >}}) 形式で直接アップロードします。インストールやセットアップなど、Prodigy の詳細については、[Prodigy のドキュメント](https://prodi.gy/docs/) を参照してください。

W&B は、画像および固有表現フィールドを [`wandb.Image`]({{< relref path="/ref/python/data-types/image" lang="ja" >}}) および [`wandb.Html`]({{< relref path="/ref/python/data-types/html" lang="ja" >}}) に自動的に変換しようとします。これらの可視化を含めるために、追加の列が結果のテーブルに追加される場合があります。

## 詳細な例を読む

W&B Prodigy integration で生成された可視化の例については、[Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) を参照してください。

## spaCy も使用していますか？

W&B には spaCy との integration もあります。[ドキュメントはこちら]({{< relref path="/guides/integrations/spacy" lang="ja" >}}) を参照してください。
