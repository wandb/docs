---
slug: /guides/integrations/yolox
description: How to integrate W&B with YOLOX.
displayed_sidebar: ja
---

# YOLOX

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)は、オブジェクト検出の性能が高いYOLOのアンカーフリーバージョンです。YOLOXには、Weights & Biasesの統合が含まれており、トレーニング、バリデーション、システムメトリクスのログ記録、およびインタラクティブなバリデーション予測を1つのコマンドライン引数で有効にできます。

## はじめに

まず、[こちら](https://wandb.ai/site)からWeights & Biasesのアカウントにサインアップしてください。

次に、`--logger wandb`コマンドライン引数を使用して、wandbでのログ記録を有効にします。オプションで、[wandb.init](../../track/launch.md)が期待するすべての引数に`wandb-`を先頭につけて渡すこともできます。

**注意:** `num_eval_imges`は、Weights & Biasesテーブルにログを記録し、モデルの評価のためにバリデーションセットの画像と予測の数を制御します。

```python
# wandbにログインします
wandb login

# `wandb` logger引数を使ってYOLOXトレーニングスクリプトを呼び出す
python tools/train.py .... --logger wandb \
                wandb-project <project-name> \
                wandb-entity <entity> \
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
```
## 例

[YOLOXのトレーニングと検証メトリクスを含む例のダッシュボードへ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

![](/images/integrations/yolox_example_dashboard.png)

このWeights & Biasesの統合に関する質問や問題がありますか？[YOLOXのGitHubリポジトリ](https://github.com/Megvii-BaseDetection/YOLOX)で問題を報告してください。確認の上、回答をお伝えします。