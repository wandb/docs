---
title: Prodigy
description: W&B を Prodigy と連携する方法
menu:
  default:
    identifier: prodigy
    parent: integrations
weight: 290
---

[Prodigy](https://prodi.gy/) は、機械学習モデルのトレーニングおよび評価用データのアノテーション、エラー分析、データの検査やクリーンアップのためのアノテーションツールです。[W&B Tables]({{< relref "/guides/models/tables/tables-walkthrough.md" >}}) を使えば、W&B 内でデータセットのログ、可視化、分析、共有（さらに多くのこと）ができます。

[W&B と Prodigy のインテグレーション](https://github.com/wandb/wandb/blob/master/wandb/integration/prodigy/prodigy.py) により、Prodigy でアノテーションしたデータセットを、簡単かつ手軽に W&B に直接アップロードし、Tables で活用できるようになります。

以下のような数行のコードを実行するだけです。

```python
import wandb
from wandb.integration.prodigy import upload_dataset

with wandb.init(project="prodigy"):
    upload_dataset("news_headlines_ner")
```

すると、次のような視覚的・インタラクティブで共有可能なテーブルが得られます。

{{< img src="/images/integrations/prodigy_interactive_visual.png" alt="Prodigy annotation table" >}}

## クイックスタート

`wandb.integration.prodigy.upload_dataset` を使えば、ローカルの Prodigy データベースからアノテーション済みの prodigy データセットを W&B に直接 [Table]({{< relref "/ref/python/sdk/data-types/table.md" >}}) フォーマットでアップロードできます。Prodigy の詳細（インストールやセットアップ含む）は [Prodigy ドキュメント](https://prodi.gy/docs/) をご参照ください。

W&B は画像や固有表現フィールドを自動的に [`wandb.Image`]({{< relref "/ref/python/sdk/data-types/image.md" >}}) や [`wandb.Html`]({{< relref "/ref/python/sdk/data-types/html.md" >}}) へ変換しようとします。これらを可視化するために追加のカラムがテーブルに加えられることがあります。

## 詳細な例を見てみましょう

W&B Prodigy インテグレーションを使って生成された可視化例は、[Visualizing Prodigy Datasets Using W&B Tables](https://wandb.ai/kshen/prodigy/reports/Visualizing-Prodigy-Datasets-Using-W-B-Tables--Vmlldzo5NDE2MTc) でご覧いただけます。

## spaCy も使っていますか？

W&B には spaCy 用のインテグレーションもあります。詳細は [こちらのドキュメント]({{< relref "/guides/integrations/spacy" >}}) をご覧ください。