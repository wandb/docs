---
slug: /guides/integrations/accelerate
description: Training and inference at scale made simple, efficient and adaptable
displayed_sidebar: default
---

# Hugging Face Accelerate

Accelerateは、わずか4行のコードを追加するだけで、同じPyTorchコードを任意の分散設定で実行できるライブラリで、シンプルかつ効率的で適応力のあるスケールでのトレーニングと推論が可能になります。

Accelerateには、Weights & Biases トラッカーが含まれており、以下でその使い方を紹介します。Accelerate トラッカーについてもっと詳しく知りたい場合は、**[こちらのドキュメント](https://huggingface.co/docs/accelerate/main/en/usage_guides/tracking)**を参照してください。

## Accelerateでのログの開始

AccelerateとWeights & Biasesを始めるには、以下の疑似コードに従ってください。

```python
from accelerate import Accelerator

# Acceleratorオブジェクトにwandbでログを取るよう指示する
accelerator = Accelerator(log_with="wandb")

# wandb runを初期化し、wandbパラメータや設定情報を渡す
accelerator.init_trackers(
    project_name="my_project", 
    config={"dropout": 0.1, "learning_rate": 1e-2}
    init_kwargs={"wandb": {"entity": "my-wandb-team"}}
    )

...

# `accelerator.log` を呼び出して wandb にログを送信し、`step` はオプションです
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=global_step)


# wandb トラッカーが正しく終了することを確認してください
accelerator.end_training()
```

詳しく説明すると、次の作業が必要です。
1. Accelerator クラスを初期化する際に、`log_with="wandb"` を渡します
2. [`init_trackers`](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.init_trackers) メソッドを呼び出し、次のものを渡します。
- `project_name` 経由でプロジェクト名
- `init_kwargs` にネストされた dict を通じて [`wandb.init`](https://docs.wandb.ai/ref/python/init) に渡したいパラメータ
- `config` を介して wandb run にログを追加したいその他の実験設定情報
3. `.log` メソッドを使って Weights & Biases にログを記録；`step`引数はオプションです
4. トレーニングが終了したら `.end_training` を呼び出します

## Accelerates の内部の W&B トラッカーにアクセスする

Accelerator.get_tracker() メソッドを使用して、すばやく wandb トラッカーにアクセスできます。トラッカーの .name 属性に対応する文字列を渡すだけで、メインプロセス上でそのトラッカーを返します。

```python
wandb_tracker = accelerator.get_tracker("wandb")
```
そこから通常通り wandb の run オブジェクトとやりとりできます。

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```
:::注意

Accelerateに組み込まれたトラッカーは、自動的に適切なプロセスで実行されるため、トラッカーがメインプロセスでのみ実行されるように設定されている場合、自動的にそうなります。



Accelerateのラッピングを完全に解除したい場合は、以下のようにして同じ結果を得ることができます。



```

wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

with accelerator.on_main_process:

    wandb_tracker.log_artifact(some_artifact_to_log)

```

:::



## Accelerateの記事

以下は、Accelerateに関する記事です。お楽しみください。



<details>



<summary>HuggingFace AccelerateがWeights & Biasesでさらにパワーアップ</summary>



* この記事では、HuggingFace Accelerateが提供する機能と、分散トレーニングや評価を行いながら、Weights & Biasesに結果をログする方法を簡単に紹介します。



詳細なレポートは[こちら](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)でご覧いただけます。

</details>