---
title: Hugging Face Accelerate
description: 大規模なトレーニングと推論をシンプルかつ効率的、柔軟に実現
menu:
  default:
    identifier: ja-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate は、同じ PyTorch コードをあらゆる分散設定で動作させることができ、大規模なモデルのトレーニングや推論をシンプルにするライブラリです。

Accelerate には W&B Tracker が含まれており、その使い方を以下で説明します。また、[Hugging Face における Accelerate Trackers](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking) についても詳しく読むことができます。

## Accelerate でログ記録を始める

Accelerate と W&B の導入方法は、以下の擬似コードの通りです。

```python
from accelerate import Accelerator

# Accelerator オブジェクトに wandb でログを取るように指示
accelerator = Accelerator(log_with="wandb")

# wandb の run を初期化し、wandb のパラメータや設定情報を渡す
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log` を呼び出して wandb にログを送信、`step` は任意
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# 最後に wandb トラッカーが正しく終了するようにする
accelerator.end_training()
```

詳細な手順は以下の通りです。
1. Accelerator クラスの初期化時に `log_with="wandb"` を渡します
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出し、以下を渡します：
  - `project_name` を使ってプロジェクト名を指定
  - [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に渡したいパラメータを、ネストした dict で `init_kwargs` に指定
  - wandb run に記録したいその他実験設定情報は `config` に指定
3. `.log` メソッドで Weights & Biases へログを記録します。`step` 引数は任意です
4. トレーニングが終了したら `.end_training` を呼び出します

## W&B トラッカーへアクセスする

W&B トラッカーへアクセスするには、`Accelerator.get_tracker()` メソッドを使用します。トラッカーの `.name` 属性に対応した文字列を渡すと、`main` プロセス上でそのトラッカーが返されます。

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
ここから先は、通常通り wandb の run オブジェクトとやり取りできます。

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate 内蔵のトラッカーは、自動的に正しいプロセス上で実行されます。トラッカーが main プロセスだけで実行されるべき場合も、自動でそのように制御されます。

Accelerate のラッピングを完全に外したい場合は、以下のように同じことが可能です。

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 関連記事
Accelerate に関するおすすめ記事を紹介します。

<details>

<summary>HuggingFace Accelerate Super Charged With W&B</summary>

* この記事では、HuggingFace Accelerate の機能や、分散トレーニング・評価をどれだけ簡単に実施できるか、そして W&B への結果の記録方法を解説します。

[Hugging Face Accelerate Super Charged with W&B レポート](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs) を読む
</details>
<br /><br />