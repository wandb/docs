---
title: Hugging Face Accelerate
description: 大規模なトレーニングと推論を、シンプルかつ効率的、柔軟に実現
menu:
  default:
    identifier: accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate は、同じ PyTorch コードをあらゆる分散設定で動作させ、モデルのトレーニングや推論を大規模に簡単に行うためのライブラリです。

Accelerate には W&B Tracker が内蔵されており、以下でその使用方法を紹介します。また、[Hugging Face ドキュメントの Accelerate Trackers](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)もご覧いただけます。

## Accelerate でログを始める

Accelerate と W&B を使い始めるには、下記の疑似コードに従ってください。

```python
from accelerate import Accelerator

# Accelerator オブジェクトに wandb でログを取るよう指定
accelerator = Accelerator(log_with="wandb")

# wandb run を初期化し、wandb のパラメータや設定情報を渡す
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log` を呼び出し、wandb にログを記録。`step` は任意
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb tracker が正しく終了するようにする
accelerator.end_training()
```

詳しく説明すると、次の手順を行う必要があります。
1. Accelerator クラスの初期化時に `log_with="wandb"` を指定します。
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出し、以下を渡します：
- `project_name` でプロジェクト名
- ネストした dict で `init_kwargs` へ [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) に渡したいパラメータ
- `config` 経由で wandb run に記録したいその他の実験設定情報
3. `.log` メソッドを使って Weights & Biases にログを記録します。`step` 引数は任意です
4. トレーニングが終了したら `.end_training` を呼び出します

## W&B トラッカーへのアクセス

W&B トラッカーへアクセスするには、`Accelerator.get_tracker()` メソッドを使用してください。トラッカーの `.name` 属性に対応する文字列を渡すことで、`main` プロセス上のトラッカーが返されます。

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
ここからは、通常通り wandb の run オブジェクトとやり取りできます。

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate に組み込まれているトラッカーは、適切なプロセスで自動的に実行されます。トラッカーが main プロセスだけで実行される仕様の場合、自動的にそのように動作します。

Accelerate のラッピングを完全に取り除きたい場合は、以下のようにすれば同じ結果が得られます。

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 関連記事
以下はおすすめの Accelerate 関連記事です。

<details>

<summary>HuggingFace Accelerate Super Charged With W&B</summary>

* この記事では、HuggingFace Accelerate の特徴や、分散トレーニングと評価をどれだけ簡単に行えるか、そして W&B で結果をログ化する方法について解説します。

[Hugging Face Accelerate Super Charged with W&B レポートを読む](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)
</details>
<br /><br />