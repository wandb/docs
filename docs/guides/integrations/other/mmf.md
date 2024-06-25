---
description: Meta AI の MMF と W&B を統合する方法
slug: /guides/integrations/mmf
displayed_sidebar: default
---


# MMF

[Meta AI's MMF](https://github.com/facebookresearch/mmf) ライブラリの `WandbLogger` クラスを使用すると、Weights & Biases でトレーニング/バリデーションのメトリクス、システム (GPU および CPU) メトリクス、モデルチェックポイント、設定パラメータをログできます。

### 現在の機能

MMFの `WandbLogger` で現在サポートされている機能は以下の通りです:

* トレーニング & バリデーションのメトリクス
* 時間に伴う学習率
* W&B Artifactsへのモデルチェックポイント保存
* GPUおよびCPUシステムメトリクス
* トレーニング設定パラメータ

### 設定パラメータ

wandb ロギングを有効にしてカスタマイズするための MMF 設定オプションは以下の通りです:

```
training:
    wandb:
        enabled: true
        
        # entityは、runを送信するユーザー名またはチーム名です。
        # デフォルトでは、runはユーザーアカウントにログされます。
        entity: null
        
        # wandbで実験をログするときに使用するプロジェクト名
        project: mmf
        
        # 実験をログする際に使用する実験/ run名
        # デフォルトの実験名は: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # モデルのチェックポイント機能を有効にし、チェックポイントを
        # W&B Artifactsに保存する
        log_model_checkpoint: true
        
        # wandb.init() に渡したい追加の引数値
        # 使用可能な引数については https://docs.wandb.ai/ref/python/init のドキュメントをチェックしてください。
        # 例えば:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandbのメタデータが保存されるディレクトリのパスを変更するには
    # (デフォルト: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```