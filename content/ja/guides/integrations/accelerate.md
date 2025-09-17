---
title: Hugging Face Accelerate
description: 大規模な トレーニング と推論を、シンプルに、効率的に、柔軟に
menu:
  default:
    identifier: ja-guides-integrations-accelerate
    parent: integrations
weight: 140
---

Hugging Face Accelerate は、同じ PyTorch コードをあらゆる分散構成で実行できるようにし、大規模なモデルトレーニングや推論を簡単にする ライブラリ です。

Accelerate には W&B Tracker が同梱されており、以下でその使い方を紹介します。詳しくは [Accelerate Trackers in Hugging Face](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking) も参照してください。

## Accelerate でログを開始する

Accelerate と W&B を使い始めるには、以下の擬似コードに従ってください。

```python
from accelerate import Accelerator

# Accelerator オブジェクトに wandb でログするよう指示する
accelerator = Accelerator(log_with="wandb")

# wandb の run を初期化し、wandb のパラメータや各種設定情報を渡す
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log` を呼び出して wandb にログする。`step` は任意
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb トラッカーが正しく終了するようにする
accelerator.end_training()
```

詳しくは、次の手順が必要です:
1. Accelerator クラスを初期化する際に `log_with="wandb"` を渡す
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出し、次を渡します:
- `project_name` でプロジェクト名を渡す
- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に渡したいパラメータは、入れ子の dict で `init_kwargs` に渡す
- `config` で wandb の run にログしたいその他の 実験 設定情報を渡す
3. `.log` メソッドで Weights & Biases にログする。`step` 引数は任意
4. トレーニングが終了したら `.end_training` を呼び出す

## W&B トラッカーにアクセスする

W&B トラッカーに アクセス するには、`Accelerator.get_tracker()` メソッドを使います。トラッカーの `.name` 属性に対応する文字列を渡すと、`main` プロセス上のそのトラッカーが返されます。

```python
wandb_tracker = accelerator.get_tracker("wandb")

```
そこからは、通常どおり wandb の run オブジェクトとやり取りできます:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

{{% alert color="secondary" %}}
Accelerate に組み込まれたトラッカーは適切なプロセスで自動的に実行されます。したがって、トラッカーが main プロセスでのみ実行される想定であれば、そのとおり自動で動作します。

Accelerate によるラッピングを完全に外したい場合は、次のようにして同じことができます:

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
{{% /alert %}}

## Accelerate 関連記事
以下はおすすめの Accelerate 記事です

<details>

<summary>HuggingFace Accelerate Super Charged With W&B</summary>

* 本記事では、HuggingFace Accelerate の提供する機能、分散トレーニングと評価をどれほど簡単に行えるか、そしてその間に W&B へ 結果 を ログ する方法を紹介します。

[Hugging Face Accelerate Super Charged with W&B Report](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs) をご覧ください。
</details>
<br /><br />