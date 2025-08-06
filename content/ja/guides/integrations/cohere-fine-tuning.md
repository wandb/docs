---
title: Cohere ファインチューニング
description: W&B を使って Cohere モデルをファインチューンする方法
menu:
  default:
    identifier: cohere-fine-tuning
    parent: integrations
weight: 40
---

W&B を使うと、Cohere モデルのファインチューニングメトリクスや設定を ログ して、モデルのパフォーマンスを分析・理解し、その結果を同僚と共有することができます。

[Cohere のガイド](https://docs.cohere.com/page/convfinqa-finetuning-wandb) にはファインチューニング run を始める方法の全体例が記載されており、[Cohere API ドキュメントはこちら](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb) で確認できます。

## Cohere のファインチューニング結果をログする

Cohere のファインチューニング ログ をあなたの W&B Workspace に追加するには:

1. W&B の APIキー、`entity` 、`project` 名を指定して `WandbConfig` を作成します。APIキーは https://wandb.ai/authorize から取得できます。

2. この設定を `FinetunedModel` オブジェクトに、モデル名、データセット、ハイパーパラメータと一緒に渡して、ファインチューニング run を開始します。

    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B の詳細情報で設定を作成
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 提供された APIキー に紐づいた有効な entity 名を指定
        project="cohere-ft",
    )

    ...  # データセットやハイパーパラメータを設定

    # cohere でファインチューニング run を開始
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

3. 作成した W&B Project 上で、モデルのファインチューニング時のトレーニング・バリデーションメトリクスやハイパーパラメータを確認できます。

    {{< img src="/images/integrations/cohere_ft.png" alt="Cohere fine-tuning dashboard" >}}


## Runs を整理する

あなたの W&B Runs は自動的に整理され、ジョブタイプやベースモデル、学習率など、どんな設定パラメータでもフィルタ・ソートできます。

さらに、run 名の変更、ノートの追加、タグ作成によるグループ化も可能です。


## リソース

* [Cohere Fine-tuning Example](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)
