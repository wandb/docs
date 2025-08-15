---
title: MMF
description: W&B を Meta AI の MMF と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-mmf
    parent: integrations
weight: 220
---

`WandbLogger` クラスは、[Meta AI の MMF](https://github.com/facebookresearch/mmf) ライブラリにおいて、W&B がトレーニング／バリデーションのメトリクス、システム（GPU および CPU）メトリクス、モデルのチェックポイント、設定パラメータをログできるようにします。

## 現在の対応機能

MMF の `WandbLogger` で現在サポートされている機能は以下の通りです:

* トレーニング & バリデーションのメトリクス
* 学習率の変化（Learning Rate over time）
* モデルのチェックポイントを W&B Artifacts へ保存
* GPU・CPU のシステムメトリクス
* トレーニング設定パラメータの記録

## 設定パラメータ

MMF の設定ファイルで利用可能な、wandb ログを有効化およびカスタマイズするためのオプションは以下の通りです:

```
training:
    wandb:
        enabled: true
        
        # entity は、Run を送信するユーザー名またはチーム名です。
        # デフォルトでは、自分のユーザーアカウントに Run が記録されます。
        entity: null
        
        # wandb で実験をログする際に使用する Project 名
        project: mmf
        
        # wandb の Project 配下で実験をログする際に使う Experiment/ run 名。
        # デフォルトの Experiment 名は: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # モデルのチェックポイント保存を有効化し、チェックポイントを W&B Artifacts に保存します
        log_model_checkpoint: true
        
        # wandb.init() に渡したい追加の引数例：
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb のメタデータを保存するディレクトリのパスを変更します（デフォルト: env.log_dir）:
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```