---
title: Prodigy
description: W&B と Prodigy を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) は、機械学習 モデル のトレーニングおよび評価用データ、エラー分析、データの検査とクレンジングのためのアノテーション ツールです。[W&B テーブル]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) を使うと、W&B 内でデータセットをログ、可視化、分析、共有（ほかにもいろいろ！）できます。

[W&B と Prodigy のインテグレーション](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) は、Prodigy でアノテーションしたデータセットを W&B に直接アップロードし、W&B テーブルで活用できるシンプルで使いやすい機能を提供します。

次のように、数行のコードを実行します。

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

すると、次のような視覚的でインタラクティブ、かつ共有可能なテーブルが得られます。

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="Prodigy のアノテーション テーブル" >}}

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使うと、ローカルの Prodigy データベースにあるアノテーション済みの Prodigy データセットを、W&B の [テーブル]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}) 形式で W&B に直接アップロードできます。Prodigy のインストールやセットアップなど、詳細は [Prodigy のドキュメント](https://prodi.gy/docs/) を参照してください。

W&B は、画像フィールドと固有表現フィールドを、それぞれ [`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) と [`wandb.Html`]({{< relref path="/ref/python/sdk/data-types/html.md" lang="ja" >}}) に自動的に変換しようとします。これらの可視化を含めるため、生成されるテーブルに追加の列が加えられる場合があります。

## 詳細な例を読む

W&B の Prodigy インテグレーションで生成された可視化の例として、[W&B Tables を使った Prodigy データセットの可視化](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) をご覧ください。  

## spaCy も使っていますか？

W&B には spaCy とのインテグレーションもあります。詳しくは [こちらのドキュメント]({{< relref path="/guides/integrations/spacy" lang="ja" >}}) をご覧ください。