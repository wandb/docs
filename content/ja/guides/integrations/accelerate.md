---
title: Hugging Face Accelerate
description: 大規模な トレーニング と推論を、シンプルかつ効率的、そして適応的に
menu:
  default:
    identifier: ja-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate は、分散構成全体で同じ PyTorch コードを実行できるようにする ライブラリ で、大規模な モデルトレーニング と推論を簡素化します。

Accelerate には、以下に示す Weights & Biases Tracker が含まれています。Accelerate Tracker の詳細については、**[こちら (英語) のドキュメント](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)** をお読みください。

## Accelerate で ログ 記録を開始する

Accelerate と Weights & Biases を使い始めるには、以下の擬似コードに従ってください。

```python
from accelerate import Accelerator

# Accelerator オブジェクトに wandb で ログ 記録するように指示します
accelerator = Accelerator(log_with="wandb")

# wandb の パラメータ と 設定 情報を渡して、wandb の run を初期化します
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log` を呼び出して wandb に ログ 記録します。`step` はオプションです
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb tracker が正しく終了することを確認します
accelerator.end_training()
```

さらに説明すると、次のことが必要です。
1. Accelerator クラス を初期化する際に `log_with="wandb"` を渡します
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッド を呼び出して、以下を渡します。
- `project_name` を介した プロジェクト 名
- ネストされた dict を介して [`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に渡したい パラメータ を `init_kwargs` に渡します
- `config` を介して wandb の run に ログ 記録したい他の 実験 設定 情報
3. `.log` メソッド を使用して Weights & Biases に ログ 記録します。`step` 引数 はオプションです
4. トレーニング 終了時に `.end_training` を呼び出します

## W&B tracker への アクセス

W&B tracker にアクセスするには、`Accelerator.get_tracker()` メソッド を使用します。tracker の `.name` 属性 に対応する文字列を渡します。これにより、`main` プロセス で tracker が返されます。

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
そこから、通常どおり wandb の run オブジェクト を操作できます。

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate に組み込まれた tracker は、正しい プロセス で自動的に実行されるため、tracker が main プロセス でのみ実行されるように設計されている場合、自動的に実行されます。

Accelerate のラッピングを完全に削除したい場合は、次の方法で同じ結果を得ることができます。

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 記事
以下は、楽しめるかもしれない Accelerate の記事です

<details>

<summary>Weights & Biases でスーパーチャージされた HuggingFace Accelerate</summary>

* この記事では、HuggingFace Accelerate が提供するものと、分散 トレーニング と評価を非常に簡単に行う方法を見ていきます。同時に、結果を Weights & Biases に ログ 記録します。

完全な レポート は[こちら](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)をお読みください。
</details>
<br /><br />
