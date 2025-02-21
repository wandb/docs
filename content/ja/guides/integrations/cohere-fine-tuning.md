---
title: Cohere fine-tuning
description: W&B を使用して Cohere モデルをファインチューンする方法。
menu:
  default:
    identifier: ja-guides-integrations-cohere-fine-tuning
    parent: integrations
weight: 40
---

Weights & Biases を使用すると、Cohere モデルのファインチューニングメトリクスや設定をログに記録し、モデルのパフォーマンスを分析・理解し、その結果を同僚と共有することができます。

この [Cohere のガイド](https://docs.cohere.com/page/convfinqa-finetuning-wandb) には、ファインチューニング run を開始する方法の完全な例があり、[Cohere API ドキュメントはこちら](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb) で見つけることができます。

## Cohere ファインチューニング結果をログする

Cohere ファインチューニングのログを W&B ワークスペースに追加するには:

1. W&B APIキー、W&B `entity` と `project` 名を使用して `WandbConfig` を作成します。あなたの W&B APIキーは https://wandb.ai/authorize で見つけることができます。

2. この設定を、モデル名、データセット、ハイパーパラメータと共に `FinetunedModel` オブジェクトに渡して、ファインチューニング run を開始します。

    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B の詳細で設定を作成
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 提供された APIキーに関連付けられた有効なエンティティである必要があります
        project="cohere-ft",
    )

    ...  # データセットとハイパーパラメーターを設定

    # cohere でファインチューニング run を開始
    cmd_r_finetune = co.finetuning.create_finetuned_model(
      request=FinetunedModel(
        name="command-r-ft",
        settings=Settings(
          base_model=...
          dataset_id=...
          hyperparameters=...
          wandb=wandb_ft_config  # ここであなたの W&B 設定を渡す
        ),
      ),
    )
    ```

3. 作成した W&B プロジェクトで、モデルのファインチューニングトレーニングとバリデーションメトリクス、ハイパーパラメータを表示します。

    {{< img src="/images/integrations/cohere_ft.png" alt="" >}}

## runs を整理する

あなたの W&B runs は自動的に整理され、ジョブタイプ、ベースモデル、学習率、その他のハイパーパラメーターなどの任意の設定パラメータに基づいてフィルタリングまたは並べ替えることができます。

さらに、runs の名前を変更したり、メモを追加したり、タグを作成してグループ化することができます。

## リソース

* **[Cohere ファインチューニング例](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)**