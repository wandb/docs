---
title: Cohere ファインチューニング
description: W&B を使って Cohere モデルをファインチューンする方法
menu:
  default:
    identifier: ja-guides-integrations-cohere-fine-tuning
    parent: integrations
weight: 40
---

W&B を使うと、Cohere モデルのファインチューニングのメトリクスや設定をログし、モデルのパフォーマンスを分析・理解したり、同僚と結果を共有したりできます。

[Cohere のこのガイド](https://docs.cohere.com/page/convfinqa-finetuning-wandb) では、ファインチューニング run を開始する方法の全体的な例が紹介されています。また、[Cohere API ドキュメントはこちら](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb) で確認できます。

## Cohere ファインチューニングの結果をログする

Cohere ファインチューニングのログを W&B Workspace に追加するには:

1. W&B の APIキー、`entity`、`project` 名を指定して `WandbConfig` を作成します。W&B の APIキーは https://wandb.ai/authorize で確認できます。

2. この設定を `FinetunedModel` オブジェクトに、モデル名やデータセット、ハイパーパラメーターとともに渡して、ファインチューニング run を開始します。

    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B 用の設定を作成します
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 提供されたAPIキーに紐づく有効な entity である必要があります
        project="cohere-ft",
    )

    ...  # データセットやハイパーパラメーターをセット

    # cohere でファインチューニング run を開始
    cmd_r_finetune = co.finetuning.create_finetuned_model(
      request=FinetunedModel(
        name="command-r-ft",
        settings=Settings(
          base_model=...
          dataset_id=...
          hyperparameters=...
          wandb=wandb_ft_config  # ここに W&B 設定を渡します
        ),
      ),
    )
    ```

3. 作成した W&B Project で、モデルのファインチューニングのトレーニング/バリデーションメトリクスやハイパーパラメーターを確認できます。

    {{< img src="/images/integrations/cohere_ft.png" alt="Cohere fine-tuning dashboard" >}}


## run を整理する

あなたの W&B run は自動的に整理されており、ジョブタイプやベースモデル、学習率、その他のハイパーパラメータなど、任意の設定パラメータによってフィルタやソートが可能です。

さらに、run の名前を変更したり、ノートやタグを追加してグループ化することもできます。


## リソース

* [Cohere Fine-tuning Example](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)