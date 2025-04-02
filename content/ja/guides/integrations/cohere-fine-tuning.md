---
title: Cohere fine-tuning
description: W&B を使用して Cohere モデル を ファインチューン する方法。
menu:
  default:
    identifier: ja-guides-integrations-cohere-fine-tuning
    parent: integrations
weight: 40
---

Weights & Biases を使用すると、Cohere モデルのファイン チューニング メトリクスと設定をログに記録して、モデルのパフォーマンスを分析および理解し、同僚と結果を共有できます。

この [Cohere のガイド](https://docs.cohere.com/page/convfinqa-finetuning-wandb) には、ファイン チューニング run を開始する方法の完全な例が記載されています。また、[Cohere API ドキュメントはこちら](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb)にあります。

## Cohere のファイン チューニング result をログに記録する

Cohere のファイン チューニング ログを W&B Workspace に追加するには:

1. W&B API キー、W&B `entity`、および `project` 名を使用して `WandbConfig` を作成します。W&B API キーは、https://wandb.ai/authorize で確認できます。

2. この設定を、モデル名、データセット、ハイパー パラメーターとともに `FinetunedModel` オブジェクトに渡して、ファイン チューニング run を開始します。

    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B の詳細を含む config を作成する
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 提供された API キーに関連付けられている有効な entity である必要があります
        project="cohere-ft",
    )

    ...  # データセットとハイパー パラメーターを設定する

    # cohere でファイン チューニング run を開始する
    cmd_r_finetune = co.finetuning.create_finetuned_model(
      request=FinetunedModel(
        name="command-r-ft",
        settings=Settings(
          base_model=...
          dataset_id=...
          hyperparameters=...
          wandb=wandb_ft_config  # ここに W&B config を渡す
        ),
      ),
    )
    ```

3. 作成した W&B project で、モデルのファイン チューニング トレーニング、検証メトリクス、およびハイパー パラメーターを表示します。

    {{< img src="/images/integrations/cohere_ft.png" alt="" >}}

## Runs を整理する

W&B の Runs は自動的に整理され、ジョブタイプ、ベース model、学習率、その他のハイパー パラメーターなどの任意の設定 parameter に基づいてフィルタリング/ソートできます。

さらに、Runs の名前を変更したり、メモを追加したり、タグを作成してグループ化したりできます。

## リソース

* **[Cohere Fine-tuning Example](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)**
