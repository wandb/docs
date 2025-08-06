---
title: Hugging Face AutoTrain
menu:
  default:
    identifier: autotrain
    parent: integrations
weight: 130
---

[Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/index) は、自然言語処理（NLP）タスク、コンピュータビジョン（CV）タスク、音声タスク、さらには表形式タスクまで、最新のモデルをノーコードでトレーニングできるツールです。

[W&B](https://wandb.com/) は Hugging Face AutoTrain と直接インテグレーションされており、実験管理や設定管理が可能です。CLI コマンドで 1 つのパラメーターを指定するだけで、すぐに実験管理が始められます。

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

ここでは LLM を数学データセットでファインチューニングし、[GSM8k ベンチマーク](https://github.com/openai/grade-school-math) の `pass@1` で SoTA の結果を出す例を紹介します。

## データセットの準備

Hugging Face AutoTrain では、CSV のカスタムデータセットに特定のフォーマットが必要です。

- 学習用ファイルには `text` カラムが必要で、そこをトレーニングで使います。最良の結果を得るには、`text` カラムのデータが `### Human: Question?### Assistant: Answer.` という形式になっている必要があります。優れた例として [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) をご覧ください。

    ただし、[MetaMathQA データセット](https://huggingface.co/datasets/meta-math/MetaMathQA) には `query`、`response`、`type` カラムがあります。まずこのデータセットを前処理し、`type` カラムを削除し、`query` と `response` カラムを結合して新たな `text` カラムを `### Human: Query?### Assistant: Response.` 形式で作成します。この処理後のデータセット [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath) をトレーニングに使用します。

## `autotrain` でトレーニングする

コマンドラインやノートブックから `autotrain` advanced を使ってトレーニングを始められます。`--log` 引数を使用し、`--log wandb` を指定して [W&B Run]({{< relref "/guides/models/track/runs/" >}}) に結果をログできます。 

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

## さらに詳しく

* [AutoTrain Advanced で実験管理が可能になりました](https://huggingface.co/blog/rishiraj/log-autotrain) by [Rishiraj Acharya](https://huggingface.co/rishiraj)
* [Hugging Face AutoTrain ドキュメント](https://huggingface.co/docs/autotrain/index)
