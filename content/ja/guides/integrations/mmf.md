---
title: MMF
description: Meta AI の MMF と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-mmf
    parent: integrations
weight: 220
---

Meta AI の [MMF](https://github.com/facebookresearch/mmf) ライブラリの `WandbLogger` クラスを使用すると、Weights & Biases でトレーニング/検証 メトリクス、システム (GPU および CPU) メトリクス、モデル チェックポイント、および設定 パラメータをログに記録できます。

## 現在の機能

MMF の `WandbLogger` では、現在、次の機能がサポートされています。

* トレーニングと検証の メトリクス
* 経時的な学習率
* W&B Artifacts へのモデル チェックポイントの保存
* GPU および CPU システム メトリクス
* トレーニング設定 パラメータ

## 設定 パラメータ

wandb ロギングを有効化およびカスタマイズするために、MMF 設定で次のオプションを使用できます。

```
training:
    wandb:
        enabled: true
        
        # エンティティは、run の送信先となる ユーザー名または Teams 名です。
        # デフォルトでは、run は ユーザー アカウントにログ記録されます。
        entity: null
        
        # wandb で 実験 をログ記録する際に使用する Project 名
        project: mmf
        
        # wandb で プロジェクト の下に 実験 をログ記録する際に使用する 実験/run 名。
        # デフォルトの 実験 名は次のとおりです: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # モデル の チェックポイント を有効にして、チェックポイント を W&B Artifacts に保存します
        log_model_checkpoint: true
        
        # wandb.init() に渡す追加の 引数 値。
        # 使用可能な 引数 (例:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        # については、/ref/python/init のドキュメントを参照してください。
        
env:
    # wandb の メタデータ が保存される ディレクトリー への パス を変更するには
    # (デフォルト: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```