---
title: Hugging Face Accelerate
description: 大規模な トレーニング と推論を簡単に効率的で適応可能なものにします。
menu:
  default:
    identifier: ja-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate は、同じ PyTorch コードを任意の分散設定で実行できるようにするライブラリで、モデルのトレーニングと推論を簡素化します。

Accelerate には Weights & Biases トラッカーが含まれており、以下でその使用方法を示します。また、Accelerate トラッカーについては **[こちらのドキュメント](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)** でさらに詳しく読むことができます。

## Accelerate でログを開始する

Accelerate と Weights & Biases を使い始めるには、以下の擬似コードに従うことができます。

```python
from accelerate import Accelerator

# Accelerator オブジェクトに wandb でログを記録するように指示する
accelerator = Accelerator(log_with="wandb")

# wandb の run を初期化し、wandb のパラメータと任意の設定情報を渡す
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log` を呼び出して wandb にログを記録する、`step` はオプション
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb トラッカーが正しく終了することを確認する
accelerator.end_training()
```

さらに説明すると、必要なことは次の通りです。
1. Accelerator クラスを初期化するときに `log_with="wandb"` を渡す
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出し、以下を渡す:
- `project_name` を介してプロジェクト名
- `init_kwargs` にネストされた dict で渡したい任意のパラメータを [`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に対して渡す
- `config` を介して wandb run にログを記録したい任意の実験設定情報
3. Weigths & Biases にログを記録するために `.log` メソッドを使用する; `step` 引数はオプションです
4. トレーニングが終了したら `.end_training` を呼び出す

## W&B トラッカーにアクセスする

W&B トラッカーにアクセスするには、`Accelerator.get_tracker()` メソッドを使用します。トラッカーの `.name` 属性に対応する文字列を渡し、`main` プロセスでトラッカーを返します。

```python
wandb_tracker = accelerator.get_tracker("wandb")

```

そこから通常通り wandb の run オブジェクトとやり取りすることができます。

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate に組み込まれたトラッカーは、正しいプロセスで自動的に実行されるので、トラッカーが主プロセスでのみ実行されるように意図されている場合、それは自動的に行われます。

Accelerate のラッピングを完全に削除したい場合は、次のようにして同じ結果を得ることができます。

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 記事
以下は、楽しめる Accelerate の記事です。

<details>

<summary>HuggingFace Accelerate Super Charged With Weights & Biases</summary>

* この記事では、HuggingFace Accelerate が提供するものと、分散トレーニングと評価を行いながら Weights & Biases に結果をログすることがいかに簡単であるかを見ていきます。

完全なレポートを [こちら](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs) で読むことができます。
</details>
<br /><br />