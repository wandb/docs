---
description: W&B を Prodigy と統合する方法
slug: /guides/integrations/prodigy
displayed_sidebar: default
---


# Prodigy

[Prodigy](https://prodi.gy/) は、機械学習モデルのトレーニングと評価用データの作成、エラー分析、データ検査 & クリーニングのためのアノテーションツールです。[W&B Tables](../../tables/tables-walkthrough.md)を使用すると、W&B 内でデータセット (およびその他) をログ、可視化、分析、および共有することができます。

[W&B integration with Prodigy](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) は、Prodigy でアノテーションを付けたデータセットを、簡単に W&B に直接アップロードして Tables で使用するための機能を追加します。

以下のようなコードを数行実行します：

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

すると、以下のような視覚的でインタラクティブかつ共有可能なテーブルが得られます：

![](/images/integrations/prodigy_interactive_visual.png)

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使用して、ローカルの Prodigy データベースから W&B の [Table](https://docs.wandb.ai/ref/python/data-types/table) 形式に直接アノテーションされた Prodigy データセットをアップロードします。インストールおよびセットアップを含む Prodigy の詳細については、[Prodigy documentation](https://prodi.gy/docs/) をご参照ください。

W&B は、自動的に画像や実体認識フィールドを [`wandb.Image`](https://docs.wandb.ai/ref/python/data-types/image) および [`wandb.Html`](https://docs.wandb.ai/ref/python/data-types/html) に変換しようとします。これらの可視化を含むために、結果のテーブルに追加の列が追加されることがあります。

## 詳細な例を読み解く

W&B Prodigy integration で生成された例の可視化を確認するには、[Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) をご覧ください。

## また、spaCy を使用していますか？

W&B には spaCy とのインテグレーションもあります。詳しくは [docs here](https://docs.wandb.ai/guides/integrations/spacy) をご覧ください。