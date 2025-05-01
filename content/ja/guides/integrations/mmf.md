---
title: MMF
description: W&B を Meta AI の MMF と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-mmf
    parent: integrations
weight: 220
---

`WandbLogger` クラスは、[Meta AI の MMF](https://github.com/facebookresearch/mmf) ライブラリで Weights & Biases を使用して、トレーニング/検証メトリクス、システム (GPU および CPU) メトリクス、モデルチェックポイント、設定パラメータをログすることを可能にします。

## 現行の機能

以下の機能は、MMF の `WandbLogger` によりサポートされています:

* トレーニング & 検証メトリクス
* 時間に応じた学習率
* モデル チェックポイントを W&B Artifacts に保存
* GPU および CPU システムメトリクス
* トレーニング設定パラメータ

## 設定パラメータ

wandb ロギングを有効にしカスタマイズするために MMF 設定で利用可能なオプションは次のとおりです:

    training:
        wandb:
            enabled: true
            
            # エンティティは、run を送信するユーザー名またはチーム名です。
            # デフォルトでは、run はユーザー アカウントにログされます。
            entity: null
            
            # wandb で実験をログする際に使用するプロジェクト名
            project: mmf
            
            # プロジェクト内で実験をログする際に使用する実験/ run 名。
            # デフォルトの実験名は: ${training.experiment_name}
            name: ${training.experiment_name}
            
            # モデル チェックポイントを有効にし、チェックポイントを W&B Artifacts に保存します
            log_model_checkpoint: true
            
            # wandb.init() に渡したい追加の引数値。
            # 使用可能な引数を確認するには、ドキュメント /ref/python/init をチェックしてください。
            # 例えば:
            # job_type: 'train'
            # tags: ['tag1', 'tag2']
            
    env:
        # wandb メタデータが保存されるディレクトリへのパスを変更するには（デフォルト: env.log_dir）:
        wandb_logdir: ${env:MMF_WANDB_LOGDIR,}