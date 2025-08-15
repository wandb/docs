---
title: Prodigy
description: W&B を Prodigy と連携する方法
menu:
  default:
    identifier: ja-guides-integrations-prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) は、機械学習モデル用のトレーニングおよび評価データの作成、エラー分析、データの検査・クレンジングのためのアノテーションツールです。[W&B Tables]({{< relref path="/guides/models/tables/tables-walkthrough.md" lang="ja" >}}) を使うことで、W&B 内でデータセットのログ、可視化、分析、共有（さらに様々なこと！）が可能になります。

[W&B の Prodigy 連携](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py)は、Prodigy でアノテーションされたデータセットを簡単に W&B にアップロードでき、Tables で利用できる機能を提供します。

以下のような数行のコードを実行するだけです:

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
# 上記コードで Prodigy でアノテーションしたデータセットをアップロードできます
```

すると、次のようなビジュアルでインタラクティブ、かつ共有可能なテーブルが手に入ります:

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="Prodigy annotation table" >}}

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使えば、ローカルの Prodigy データベースからアノテーション済みの Prodigy データセットを、W&B の [Table]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}) 形式で直接アップロードできます。Prodigy のインストールやセットアップなど、詳細については [Prodigy ドキュメント](https://prodi.gy/docs/) をご覧ください。

W&B は自動的に画像や固有表現フィールドを [`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) や [`wandb.Html`]({{< relref path="/ref/python/sdk/data-types/html.md" lang="ja" >}}) に変換しようとします。結果のテーブルには、これらの可視化を含める追加カラムが付与される場合もあります。

## 詳細な例を読んでみよう

W&B Prodigy 連携で生成された可視化の例は、[Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) でご覧いただけます。  

## spaCy もお使いですか？

W&B は spaCy とのインテグレーション機能も提供しています。[こちらのドキュメント]({{< relref path="/guides/integrations/spacy" lang="ja" >}})をご覧ください。