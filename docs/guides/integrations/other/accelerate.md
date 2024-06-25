---
description: 大規模なトレーニングと推論をシンプル、効率的、かつ適応可能にします
slug: /guides/integrations/accelerate
displayed_sidebar: default
---


# Hugging Face Accelerate

Accelerateは、同じPyTorchコードを任意の分散設定で実行できるようにするライブラリで、わずか4行のコードを追加するだけで、スケールされたトレーニングと推論を簡単、効率的かつ適応可能にします。

AccelerateにはWeights & Biases Trackerが含まれており、以下でその使用方法を示します。また、Accelerate Trackersについて詳しくは[こちらのドキュメント](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)をご覧ください。

## Accelerateでログを開始する

AccelerateとWeights & Biasesを使い始めるには、以下の擬似コードに従ってください:

```python
from accelerate import Accelerator

# Acceleratorオブジェクトにwandbでログを記録するように指示
accelerator = Accelerator(log_with="wandb")

# wandb runを初期化し、wandbパラメータと設定情報を渡す
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2},
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log`を呼び出してwandbにログを記録、`step`はオプション
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb trackerを正しく終了させる
accelerator.end_training()
```

さらに詳しく説明すると、以下が必要です:
1. Acceleratorクラスを初期化する際に `log_with="wandb"` を渡す
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出し、以下を渡す:
   - `project_name` を通じてプロジェクト名
   - ネストされた辞書で `init_kwargs` に渡す Any パラメータ: [`wandb.init`](https://docs.wandb.ai/ref/python/init)
   - `config` を通じて wandb run にログを記録したいその他の実験設定情報
3. `.log` メソッドを使用してWeights & Biasesにログを記録する; `step`引数はオプション
4. トレーニングが終了したら `.end_training` を呼び出す

## Accelerateの内部W&B Trackerへのアクセス

Accelerator.get_tracker() メソッドを使用すると、素早くwandbトラッカーにアクセスできます。トラッカーの `.name` 属性に対応する文字列を渡すだけで、そのトラッカーをメインプロセス上で返します。

```python
wandb_tracker = accelerator.get_tracker("wandb")
```

そこから、通常通りwandbのrunオブジェクトと対話できます:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

:::caution
Accelerateに組み込まれたトラッカーは、自動的に正しいプロセスで実行されるため、トラッカーがメインプロセスでのみ実行されるべき場合は、自動的にそうなります。

Accelerateのラッピングを完全に削除したい場合、次のようにすることで同じ結果が得られます:

```
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
with accelerator.on_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
:::

## Accelerate Articles
以下はAccelerateに関する記事です、お楽しみください

<details>

<summary>HuggingFace Accelerate Super Charged With Weights & Biases</summary>

* この記事では、HuggingFace Accelerateの提供内容と、分散トレーニングや評価を行いながら、結果をWeights & Biasesにログする方法の簡単さについて紹介します。

全レポートを[こちら](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)からお読みください。
</details>