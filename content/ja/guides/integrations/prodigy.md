---
title: プロディジー
description: W&B を Prodigy と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) は、機械学習モデルのトレーニングと評価用データを作成するためのアノテーションツールであり、エラー分析、データの調査とクリーニングにも使用されます。[W&B Tables]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) を使用すると、W&B 内でデータセットのログ、可視化、分析、共有（およびそれ以上！）が可能です。

[W&B の Prodigy とのインテグレーション](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) により、Prodigy でアノテーションされたデータセットを W&B に直接アップロードし、Tables と一緒に使用するシンプルで使いやすい機能が追加されます。

このようにいくつかのコードを実行します：

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

すると、このような視覚的でインタラクティブな共有可能なテーブルが得られます:

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="" >}}

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使用して、アノテーション済みの prodigy データセットをローカルの Prodigy データベースから直接 W&B の [Table]({{< relref path="/ref/python/data-types/table" lang="ja" >}})形式でアップロードできます。Prodigy の詳細、インストール & セットアップを含む情報は、[Prodigy ドキュメント](https://prodi.gy/docs/)を参照してください。

W&B は自動的に画像と固有表現フィールドをそれぞれ [`wandb.Image`]({{< relref path="/ref/python/data-types/image" lang="ja" >}}) と [`wandb.Html`]({{< relref path="/ref/python/data-types/html" lang="ja" >}}) に変換しようとします。これらの可視化を含めるために、結果のテーブルに追加の列が追加されることがあります。

## 詳細な例を読む

W&B Prodigy インテグレーションを使用して生成された可視化例については、[Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) を参照してください。

## spaCy も使用していますか？

W&B は spaCy とのインテグレーションも備えています。[ドキュメントはこちら]({{< relref path="/guides/integrations/spacy" lang="ja" >}})を参照してください。