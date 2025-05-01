---
title: Hugging Face Accelerate
description: 大規模なトレーニングと推論がシンプルで効率的、かつ適応可能に
menu:
  default:
    identifier: ja-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate は、同じ PyTorch コードを任意の分散設定で実行できるようにするライブラリで、モデルトレーニングとスケールでの推論を簡素化します。

Accelerate は Weights & Biases Tracker を含んでおり、以下でその使用方法を示します。また、Accelerate Trackers について詳しくは **[こちらのドキュメント](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)** をご覧ください。

## Accelerate を使ってログを開始する

Accelerate と Weights & Biases を使用するには、以下の疑似コードに従ってください。

```python
from accelerate import Accelerator

# Accelerator オブジェクトに wandb でログを記録するように伝える
accelerator = Accelerator(log_with="wandb")

# wandb run を初期化し、wandb のパラメータと任意の設定情報を渡す
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log`を呼び出して wandb にログを記録する、`step` はオプション
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb トラッカーが正しく終了するようにする
accelerator.end_training()
```

さらに説明すると、以下の手順が必要です。
1. Accelerator クラスを初期化するときに `log_with="wandb"` を渡す
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出し、以下を渡します:
- `project_name` よりプロジェクト名
- [`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に渡したい任意のパラメータをネストされた dict で `init_kwargs` に
- wandb run にログ記録したい任意の実験設定情報を `config` で
3. Weights & Biases にログを記録するために `.log` メソッドを使用する; `step` 引数はオプション
4. トレーニングが終了したら `.end_training` を呼び出す

## W&B トラッカーへのアクセス

W&B トラッカーにアクセスするには、`Accelerator.get_tracker()` メソッドを使用します。トラッカーの`.name`属性に対応する文字列を渡すと、`main` プロセスのトラッカーが返されます。

```python
wandb_tracker = accelerator.get_tracker("wandb")

```

そこから、通常通り wandb の run オブジェクトと対話できます:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate にビルトインされたトラッカーは、正しいプロセスで自動的に実行されるので、トラッカーがメインプロセスでのみ実行するように設定されている場合、それが自動的に行われます。

Accelerate のラッピングを完全に削除したい場合は、次の方法で同じ結果を得ることができます:

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 記事
以下は Accelerate 記事で、お楽しみいただけるかもしれません。

<details>

<summary>HuggingFace Accelerate Super Charged With Weights & Biases</summary>

* この記事では、HuggingFace Accelerate が提供するものと、Weights & Biases に結果を記録しながら分散トレーニングと評価を簡単に行う方法を紹介します。

完全なレポートは [こちら](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)をご覧ください。
</details>
<br /><br />