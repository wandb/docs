---
description: W&B を YOLOX と統合する方法
slug: /guides/integrations/yolox
displayed_sidebar: default
---


# YOLOX

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) は、オブジェクト検出のパフォーマンスが高いアンカーフリーのバージョンのYOLOです。YOLOXにはWeights & Biasesのインテグレーションが含まれており、1つのコマンドライン引数でトレーニング、検証、システムメトリクスのログ、およびインタラクティブな検証予測を有効にすることができます。

## はじめに

YOLOXをWeights & Biasesで使用するには、まず[こちら](https://wandb.ai/site)でWeights & Biasesアカウントにサインアップする必要があります。

次に、`--logger wandb` コマンドライン引数を使用してwandbでのログ記録を有効にします。オプションとして、[wandb.init](../../track/launch.md) が期待するすべての引数を渡すこともできます。その際、各引数の先頭に `wandb-` を付けてください。

**注意:** `num_eval_imges` は検証セットの画像と、モデルの評価のためにWeights & Biases Tablesにログされる予測の数を制御します。

```shell
# wandbにログイン
wandb login

# `wandb` ロガー引数を使用してyoloxトレーニングスクリプトを呼び出します
python tools/train.py .... --logger wandb \
                wandb-project <project-name> \
                wandb-entity <entity>
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
```

## 例

[YOLOXのトレーニングと検証メトリクスを示すダッシュボードの例 ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

![](/images/integrations/yolox_example_dashboard.png)

Weights & Biasesインテグレーションに関する質問や問題がある場合は、[YOLOXのGithubリポジトリ](https://github.com/Megvii-BaseDetection/YOLOX)にissueを作成してください。私たちがキャッチして回答します :)