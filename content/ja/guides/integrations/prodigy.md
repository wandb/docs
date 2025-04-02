---
title: Prodigy
description: W&B と Prodigy を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) は、機械学習モデルのトレーニング および 評価 データ、エラー分析、データ の検査 および クリーニングを作成するためのアノテーション ツールです。[W&B テーブル]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) を使用すると、W&B 内でデータセット (など) をログ記録、可視化、分析、および共有できます。

[Prodigy との W&B インテグレーション](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) は、Prodigy でアノテーションを付けたデータセットを W&B に直接アップロードして テーブル で使用するための、シンプルで使いやすい機能を追加します。

次のようなコードを数行実行します。

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

次のような、視覚的でインタラクティブな共有可能な テーブル を取得します。

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="" >}}

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使用して、アノテーションが付けられた Prodigy データセットをローカルの Prodigy データベースから W&B の [テーブル]({{< relref path="/ref/python/data-types/table" lang="ja" >}}) 形式で直接アップロードします。インストール および セットアップを含む Prodigy の詳細については、[Prodigy ドキュメント](https://prodi.gy/docs/) を参照してください。

W&B は、画像 および 固有表現フィールドを [`wandb.Image`]({{< relref path="/ref/python/data-types/image" lang="ja" >}}) および [`wandb.Html`]({{< relref path="/ref/python/data-types/html" lang="ja" >}}) にそれぞれ自動的に変換しようとします。これらの可視化を含めるために、追加の列が結果のテーブルに追加される場合があります。

## 詳細な例を読む

W&B Prodigy インテグレーションで生成された可視化の例については、[W&B テーブル を使用した Prodigy データセットの可視化](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) を参照してください。

## spaCy も使用しますか？

W&B には spaCy との インテグレーション もあります。[ドキュメントはこちら]({{< relref path="/guides/integrations/spacy" lang="ja" >}}) をご覧ください。
