---
title: Cohere fine-tuning
description: W&B を使用して Cohere モデル を ファインチューン する方法。
menu:
  default:
    identifier: ja-guides-integrations-cohere-fine-tuning
    parent: integrations
weight: 40
---

Weights & Biases を使用すると、Cohere モデルの fine-tuning メトリクスと設定をログに記録して、モデルのパフォーマンスを分析および理解し、同僚と結果を共有できます。

この [Cohere のガイド](https://docs.cohere.com/page/convfinqa-finetuning-wandb) には、fine-tuning の run を開始する方法の完全な例が記載されています。また、[Cohere API ドキュメントはこちら](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb) にあります。

## Cohere fine-tuning の結果をログに記録する

Cohere fine-tuning のログ記録を W&B の Workspace に追加するには:

1. W&B API キー、W&B の `entity` および `project` 名を使用して `WandbConfig` を作成します。W&B API キーは https://wandb.ai/authorize で確認できます。

2. この設定を、モデル名、データセット、ハイパーパラメーターとともに `FinetunedModel` オブジェクトに渡して、fine-tuning の run を開始します。

    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B の詳細を含む設定を作成する
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 提供された API キーに関連付けられている有効な entity である必要があります
        project="cohere-ft",
    )

    ...  # データセットとハイパーパラメーターを設定する

    # cohere で fine-tuning の run を開始する
    cmd_r_finetune = co.finetuning.create_finetuned_model(
      request=FinetunedModel(
        name="command-r-ft",
        settings=Settings(
          base_model=...
          dataset_id=...
          hyperparameters=...
          wandb=wandb_ft_config  # ここに W&B 設定を渡す
        ),
      ),
    )
    ```

3. 作成した W&B の project で、モデルの fine-tuning のトレーニングと検証のメトリクスおよびハイパーパラメーターを表示します。

    {{< img src="/images/integrations/cohere_ft.png" alt="" >}}

## Runs を整理する

W&B の Runs は自動的に整理され、ジョブタイプ、ベースモデル、学習率、その他のハイパーパラメーターなどの任意の設定パラメーターに基づいて、フィルタリング/ソートできます。

さらに、Runs の名前を変更したり、メモを追加したり、タグを作成してグループ化したりできます。

## リソース

*   **[Cohere Fine-tuning の例](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)**
