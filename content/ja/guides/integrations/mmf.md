---
title: MMF
description: W&B を Meta AI の MMF と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-mmf
    parent: integrations
weight: 220
---

`WandbLogger` クラスは、[Meta AI の MMF](https://github.com/facebookresearch/mmf) ライブラリにおいて、Weights & Biases がトレーニング/検証メトリクス、システム (GPU と CPU) メトリクス、モデルチェックポイント、設定パラメータをログするのを可能にします。

## 現在の機能

MMF の `WandbLogger` が現在サポートしている機能は以下の通りです:

* トレーニング & 検証メトリクス
* 時間に伴う学習率
* モデルチェックポイントの W&B Artifacts への保存
* GPU と CPU のシステムメトリクス
* トレーニング設定パラメータ

## 設定パラメータ

wandb ログを有効化しカスタマイズするために、MMF 設定で以下のオプションが利用可能です:

```
training:
    wandb:
        enabled: true
        
        # entity は、run を送信するユーザー名またはチーム名です。
        # デフォルトでは、ユーザーアカウントに run をログします。
        entity: null
        
        # wandb で実験をログする際に使用するプロジェクト名
        project: mmf
        
        # 実験/ run 名を、wandb のプロジェクト下でログする際に使用
        # デフォルトの実験名は: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # モデルのチェックポイント作成を有効にし、チェックポイントを W&B Artifacts に保存
        log_model_checkpoint: true
        
        # wandb.init() に渡したい追加の引数の値。
        # 使用可能な引数については、/ref/python/init のドキュメントを参照してください。
        # 例:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # デフォルトのパスを変更するには、wandb メタデータが保存されるディレクトリーのパスを
    # (デフォルト: env.log_dir) 変更してください。:
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```