---
title: MMF
description: W&B を Meta AI の MMF と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-mmf
    parent: integrations
weight: 220
---

Meta AI の MMF ライブラリの `WandbLogger` クラスを使用すると、Weights & Biases でトレーニング / 検証のメトリクス、システム (GPU および CPU) のメトリクス、モデル のチェックポイント、および設定 パラメータを記録できます。

## 現在の機能

現在、MMF の `WandbLogger` でサポートされている機能は次のとおりです。

* トレーニングと検証のメトリクス
* 学習率の推移
* W&B Artifacts へのモデル チェックポイントの保存
* GPU および CPU システムのメトリクス
* トレーニング設定 パラメータ

## 設定パラメータ

wandb ログを有効化およびカスタマイズするために、MMF 設定では次のオプションを使用できます。

```
training:
    wandb:
        enabled: true
        
        # エンティティは、run の送信先のユーザー名または Teams 名です。
        # デフォルトでは、run は自分のユーザーアカウントに記録されます。
        entity: null
        
        # wandb で experiment を記録する際に使用する Project 名
        project: mmf
        
        # wandb で Project の下に experiment を記録する際に使用する
        # experiment / run 名。デフォルトの experiment 名は次のとおりです。
        # ${training.experiment_name}
        name: ${training.experiment_name}
        
        # モデル のチェックポイントをオンにして、チェックポイントを W&B Artifacts に保存します
        log_model_checkpoint: true
        
        # wandb.init() に渡す追加の 引数 の 値 。
        # 引数 については、/ref/python/init のドキュメントを
        # 参照してください。たとえば、次のものがあります。
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb のメタデータが保存される ディレクトリー へのパスを変更するには
    # (デフォルト: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```