---
title: Hugging Face Accelerate
description: 大規模なトレーニングと推論を、シンプル、効率的、かつ適応的に
menu:
  default:
    identifier: ja-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate は、あらゆる分散設定で同じ PyTorch コードを実行できるようにし、大規模なモデルトレーニングと推論を簡素化するライブラリです。

Accelerate には、以下に示す Weights & Biases Tracker が含まれています。Accelerate Trackers の詳細については、**[こちら (https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking) のドキュメント]** を参照してください。

## Accelerate でログ記録を開始する

Accelerate と Weights & Biases を使い始めるには、以下の疑似コードに従ってください。

```python
from accelerate import Accelerator

# Tell the Accelerator object to log with wandb
accelerator = Accelerator(log_with="wandb")

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# Log to wandb by calling `accelerator.log`, `step` is optional
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# Make sure that the wandb tracker finishes correctly
accelerator.end_training()
```

さらに詳しく説明すると、次のことが必要です。
1. Accelerator クラスを初期化するときに `log_with="wandb"` を渡します。
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出して、以下を渡します。
- `project_name` を介したプロジェクト名
- ネストされた dict を介して [`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に渡すパラメータ `init_kwargs`
- `config` を介して、wandb の run に記録するその他の実験設定情報
3. `.log` メソッドを使用して Weigths & Biases にログを記録します。`step` 引数はオプションです。
4. トレーニングが終了したら、`.end_training` を呼び出します。

## W&B tracker へのアクセス

W&B tracker にアクセスするには、`Accelerator.get_tracker()` メソッドを使用します。tracker の `.name` 属性に対応する文字列を渡すと、`main` プロセスの tracker が返されます。

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
そこから、通常どおり wandb の run オブジェクトを操作できます。

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate に組み込まれた Trackers は、正しいプロセスで自動的に実行されるため、Tracker がメインプロセスでのみ実行されるように設計されている場合、自動的に実行されます。

Accelerate のラッピングを完全に削除したい場合は、次の方法で同じ結果を得ることができます。

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate の記事
以下は、楽しめるかもしれない Accelerate の記事です。

<details>

<summary>Weights & Biases でスーパーチャージされた HuggingFace Accelerate</summary>

* この記事では、HuggingFace Accelerate が提供するものと、結果を Weights & Biases に記録しながら、分散トレーニングと評価をどれだけ簡単に行えるかを見ていきます。

完全なレポートは [こちら](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs) をお読みください。
</details>
<br /><br />
