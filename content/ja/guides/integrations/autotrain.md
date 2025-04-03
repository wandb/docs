---
title: Hugging Face AutoTrain
menu:
  default:
    identifier: ja-guides-integrations-autotrain
    parent: integrations
weight: 130
---

[Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/index) は、自然言語処理（NLP）タスク、コンピュータビジョン（CV）タスク、音声タスク、さらには表形式タスクのために、最先端のモデルをトレーニングするためのノーコード ツールです。

[Weights & Biases](http://wandb.com/) は Hugging Face AutoTrain に直接統合されており、実験管理と構成管理を提供します。これは、実験の CLI コマンドで単一の パラメータ を使用するのと同じくらい簡単です。

{{< img src="/images/integrations/hf-autotrain-1.png" alt="実験のメトリクスを記録する例" >}}

## 前提条件のインストール

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

これらの変更を示すために、このページでは、[GSM8k Benchmarks](https://github.com/openai/grade-school-math) で `pass@1` の SoTA 結果を達成するために、数学 データセット で LLM を微調整します。

## データセット を準備する

Hugging Face AutoTrain は、適切に機能するために、CSV カスタム データセット が特定の形式になっていることを想定しています。

- トレーニング ファイルには、トレーニングで使用する `text` 列が含まれている必要があります。最良の結果を得るには、`text` 列のデータが `### Human: Question?### Assistant: Answer.` 形式に準拠している必要があります。[`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) で素晴らしい例を確認してください。

    ただし、[MetaMathQA dataset](https://huggingface.co/datasets/meta-math/MetaMathQA) には、`query`、`response`、および `type` 列が含まれています。最初に、この データセット を前処理します。`type` 列を削除し、`query` 列と `response` 列の内容を `### Human: Query?### Assistant: Response.` 形式の新しい `text` 列に結合します。トレーニングでは、結果の データセット [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath) を使用します。

## `autotrain` を使用してトレーニングする

コマンドラインまたは ノートブック から `autotrain` advanced を使用してトレーニングを開始できます。`--log` 引数を使用するか、`--log wandb` を使用して、[W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) に 結果 を ログ します。

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
# ハイパーパラメータ を設定する
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

# トレーニングを実行する
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


{{< img src="/images/integrations/hf-autotrain-2.gif" alt="実験の構成を保存する例" >}}

## その他のリソース

* [AutoTrain Advanced now supports Experiment Tracking](https://huggingface.co/blog/rishiraj/log-autotrain) by [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [Hugging Face AutoTrain Docs](https://huggingface.co/docs/autotrain/index)
