---
title: MMF
description: W&B を Meta AI の MMF と統合する方法
menu:
  default:
    identifier: mmf
    parent: integrations
weight: 220
---

`WandbLogger` クラスは、[Meta AI の MMF](https://github.com/facebookresearch/mmf) ライブラリで W&B にトレーニング/バリデーションのメトリクス、システム（GPU および CPU）メトリクス、モデルのチェックポイントや設定パラメータをログすることを可能にします。

## 現在の機能

MMF の `WandbLogger` で現在サポートされている機能は以下の通りです。

* トレーニング & バリデーションのメトリクス
* 時間に応じた学習率
* モデルのチェックポイントを W&B Artifacts へ保存
* GPU・CPU のシステムメトリクス
* トレーニング設定パラメータ

## 設定パラメータ

wandb ロギングを有効にしカスタマイズするために、MMF の設定で利用できるオプションは次の通りです。

```
training:
    wandb:
        enabled: true
        
        # entity は run を送信するユーザー名またはチーム名です。
        # デフォルトでは run はあなたのユーザーアカウントに記録されます。
        entity: null
        
        # wandb で実験を記録する際に使用される Project 名
        project: mmf
        
        # wandb で Project 配下に記録する実験/run 名
        # デフォルトの実験名は: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # モデルチェックポイントの保存を有効化し、チェックポイントを W&B Artifacts に保存します
        log_model_checkpoint: true
        
        # wandb.init() に追加で引数を渡したい場合に利用できます（例）:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb のメタデータを保存するディレクトリーのパスを変更する場合（デフォルト: env.log_dir）:
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```