---
slug: /guides/integrations/mmf
description: W&BをMeta AIのMMFと統合する方法。
---

# MMF

[Meta AIのMMF](https://github.com/facebookresearch/mmf)ライブラリの`WandbLogger`クラスを使用すると、トレーニング/検証メトリクス、システム（GPUおよびCPU）メトリクス、モデルチェックポイント、および設定パラメータをWeights＆Biasesにログできます。

### 現在の機能

MMFの`WandbLogger`で現在サポートされている機能は次のとおりです。

* トレーニング＆検証メトリクス
* 時間経過による学習率
* モデルチェックポイントをW＆Bアーティファクトに保存
* GPUおよびCPUシステムメトリクス
* トレーニング設定パラメータ

### 設定パラメータ

MMF設定でwandbロギングを有効化およびカスタマイズするための以下のオプションが利用可能です。

```
training:
    wandb:
        enabled: true
        
        # エンティティは、ランを送信しているユーザー名またはチーム名です。
        # デフォルトでは、ランがユーザーアカウントにログされます。
        entity: null

# wandbを使って実験をログに記録する際のプロジェクト名

        project: mmf

        

        # プロジェクト内で実験をログに記録する際に使用される実験/ Run名

        # デフォルトの実験名は: ${training.experiment_name}

        name: ${training.experiment_name}

        

        # モデルのチェックポイントをオンにし、チェックポイントをW＆Bアーティファクトに保存します。

        log_model_checkpoint: true

        

        # wandb.init（）に渡したい追加の引数値。

        # https://docs.wandb.ai/ref/python/init のドキュメントを調べると

        # 利用可能な引数が表示されます。例：

        # job_type: 'train'

        # tags: ['tag1', 'tag2']

        

env:

    # wandbメタデータが保存されるディレクトリーへのパスを変更するには（デフォルト: env.log_dir）：

    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}

```