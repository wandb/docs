---
title: Cohere の ファインチューニング
description: W&B で Cohere の モデルを ファインチューンする 方法。
menu:
  default:
    identifier: ja-guides-integrations-cohere-fine-tuning
    parent: integrations
weight: 40
---

W&B を使えば、Cohere モデルのファインチューニングのメトリクスや設定をログして、モデルの性能を分析・理解し、その結果を同僚と共有できます。

この [Cohere のガイド](https://docs.cohere.com/page/convfinqa-finetuning-wandb) では、ファインチューニング run の始め方が一通り示されています。あわせて [Cohere の API ドキュメントはこちら](https://docs.cohere.com/reference/createfinetunedmodel#request.body.settings.wandb) を参照してください。

## Cohere のファインチューニング結果をログする

W&B Workspace に Cohere のファインチューニングのログを追加するには:

1. W&B の APIキー、W&B `entity` と `project` 名を含む `WandbConfig` を作成します。W&B の APIキーは https://wandb.ai/authorize で取得できます。

2. この設定を `FinetunedModel` オブジェクトに、モデル名、データセット、ハイパーパラメーターと一緒に渡して、ファインチューニング run を開始します。


    ```python
    from cohere.finetuning import WandbConfig, FinetunedModel

    # W&B の情報で設定を作成する
    wandb_ft_config = WandbConfig(
        api_key="<wandb_api_key>",
        entity="my-entity", # 指定した APIキー に関連付けられた有効な entity である必要があります
        project="cohere-ft",
    )

    ...  # データセットとハイパーパラメーターを準備する

    # Cohere でファインチューニング run を開始する
    cmd_r_finetune = co.finetuning.create_finetuned_model(
      request=FinetunedModel(
        name="command-r-ft",
        settings=Settings(
          base_model=...
          dataset_id=...
          hyperparameters=...
          wandb=wandb_ft_config  # ここに W&B の設定を渡す
        ),
      ),
    )
    ```

3. 作成した W&B Project 内で、モデルのファインチューニングのトレーニングおよび検証メトリクス、ハイパーパラメーターを確認します。

    {{< img src="/images/integrations/cohere_ft.png" alt="Cohere のファインチューニング ダッシュボード" >}}


## runs を整理する

W&B の runs は自動で整理され、ジョブタイプ、ベースモデル、学習率、その他のあらゆるハイパーパラメーターなどの任意の設定パラメータに基づいて、フィルターやソートができます。

さらに、runs の名前変更、ノートの追加、タグの作成によるグルーピングも可能です。


## リソース

* [Cohere のファインチューニング例](https://github.com/cohere-ai/notebooks/blob/kkt_ft_cookbooks/notebooks/finetuning/convfinqa_finetuning_wandb.ipynb)