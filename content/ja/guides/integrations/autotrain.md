---
title: Hugging Face AutoTrain
menu:
  default:
    identifier: ja-guides-integrations-autotrain
    parent: integrations
weight: 130
---

[Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/index) は、自然言語処理（NLP）タスク、コンピュータビジョン（CV）タスク、音声タスク、さらにはテーブルデータに対しても最新のモデルをノーコードでトレーニングできるツールです。

[W&B](https://wandb.com/) は Hugging Face AutoTrain に直接インテグレーションされており、実験管理や設定管理が可能です。CLI コマンドでたった 1 つのパラメータを指定するだけで簡単に利用できます。

{{< img src="/images/integrations/hf-autotrain-1.png" alt="Experiment metrics logging" >}}

## 必要なパッケージのインストール

`autotrain-advanced` と `wandb` をインストールします。

{{< tabpane text=true >}}

{{% tab header="コマンドライン" value="script" %}}

```shell
pip install --upgrade autotrain-advanced wandb
```

{{% /tab %}}

{{% tab header="ノートブック" value="notebook" %}}

```notebook
!pip install --upgrade autotrain-advanced wandb
```

{{% /tab %}}

{{< /tabpane >}}

このページでは、これらの設定を使って LLM を数学データセットでファインチューニングし、[GSM8k Benchmarks](https://github.com/openai/grade-school-math) の `pass@1` で最先端（SoTA）の結果を目指します。

## データセットの準備

Hugging Face AutoTrain では、CSV 形式のカスタムデータセットが特定のフォーマットで整形されている必要があります。

- トレーニング用ファイルには `text` カラムが必要で、トレーニングにはこのカラムが使用されます。最良の結果を得るために、`text` カラムのデータは `### Human: 質問?### Assistant: 回答.` という形式に従う必要があります。 [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) に優れた例があります。

    ただし、[MetaMathQA データセット](https://huggingface.co/datasets/meta-math/MetaMathQA) には `query`、`response`、`type` カラムが含まれています。まずこのデータセットを前処理しましょう。`type` カラムを削除し、`query` と `response` の内容を結合して、新しい `text` カラムを `### Human: Query?### Assistant: Response.` という形式にしてください。その後、作成されたデータセット [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath) をトレーニングに使います。

## `autotrain` でトレーニング

コマンドラインまたはノートブック上で `autotrain` advanced を使ってトレーニングを始められます。`--log` 引数を利用するか、`--log wandb` を指定すると [W&B Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) に結果をログできます。

{{< tabpane text=true >}}

{{% tab header="コマンドライン" value="script" %}}

```shell
autotrain llm \
    --train \
    --model HuggingFaceH4/zephyr-7b-alpha \
    --project-name zephyr-math \
    --log wandb \
    --data-path data/ \
    --text-column text \
    --lr 2e-5 \
    --batch-size 4 \
    --epochs 3 \
    --block-size 1024 \
    --warmup-ratio 0.03 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --weight-decay 0.0 \
    --gradient-accumulation 4 \
    --logging_steps 10 \
    --fp16 \
    --use-peft \
    --use-int4 \
    --merge-adapter \
    --push-to-hub \
    --token <huggingface-token> \
    --repo-id <huggingface-repository-address>
```

{{% /tab %}}

{{% tab header="ノートブック" value="notebook" %}}

```notebook
# ハイパーパラメーターの設定
learning_rate = 2e-5
num_epochs = 3
batch_size = 4
block_size = 1024
trainer = "sft"
warmup_ratio = 0.03
weight_decay = 0.
gradient_accumulation = 4
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
logging_steps = 10

# トレーニングの実行
!autotrain llm \
    --train \
    --model "HuggingFaceH4/zephyr-7b-alpha" \
    --project-name "zephyr-math" \
    --log "wandb" \
    --data-path data/ \
    --text-column text \
    --lr str(learning_rate) \
    --batch-size str(batch_size) \
    --epochs str(num_epochs) \
    --block-size str(block_size) \
    --warmup-ratio str(warmup_ratio) \
    --lora-r str(lora_r) \
    --lora-alpha str(lora_alpha) \
    --lora-dropout str(lora_dropout) \
    --weight-decay str(weight_decay) \
    --gradient-accumulation str(gradient_accumulation) \
    --logging-steps str(logging_steps) \
    --fp16 \
    --use-peft \
    --use-int4 \
    --merge-adapter \
    --push-to-hub \
    --token str(hf_token) \
    --repo-id "rishiraj/zephyr-math"
```

{{% /tab %}}

{{< /tabpane >}}

{{< img src="/images/integrations/hf-autotrain-2.gif" alt="Experiment config saving" >}}

## さらに詳しい情報

* [AutoTrain Advanced now supports Experiment Tracking](https://huggingface.co/blog/rishiraj/log-autotrain) （[Rishiraj Acharya](https://huggingface.co/rishiraj)によるブログ）
* [Hugging Face AutoTrain Docs](https://huggingface.co/docs/autotrain/index)
