---
title: MMF
description: W&B を Meta AI の MMF と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-mmf
    parent: integrations
weight: 220
---

[Meta AI の MMF](https://github.com/facebookresearch/mmf) ライブラリの `WandbLogger` クラスは、W&B で トレーニング/検証 のメトリクス、システム (GPU と CPU) のメトリクス、モデルのチェックポイント、設定パラメータを ログ できるようにします。

## 現在の機能

MMF の `WandbLogger` では、現在 次の機能をサポートしています:

* トレーニングと検証のメトリクス
* 学習率の推移
* モデルのチェックポイントを W&B Artifacts に保存
* GPU と CPU のシステム メトリクス
* トレーニングの設定パラメータ

## 設定パラメータ

W&B ロギングを有効化してカスタマイズするために、MMF の設定で次のオプションが利用できます:

```
training:
    wandb:
        enabled: true
        
        # entity は、run を送信する宛先の ユーザー名 または team 名です。
        # 既定では、その run はあなたの user アカウントに ログ されます。
        entity: null
        
        # W&B で experiment を ログ するときに使用する project 名
        project: mmf
        
        # experiment を ログ するときに使用する experiment / run 名
        # W&B の project 配下。既定の experiment 名
        # は: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # モデルのチェックポイントを有効化し、チェックポイントを W&B Artifacts に保存します
        log_model_checkpoint: true
        
        # wandb.init() に渡したい追加の引数の例:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # W&B のメタデータを保存するディレクトリーのパスを変更します (デフォルト: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```