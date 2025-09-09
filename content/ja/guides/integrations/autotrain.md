---
title: Hugging Face AutoTrain
menu:
  default:
    identifier: ja-guides-integrations-autotrain
    parent: integrations
weight: 130
---

[Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/index) は、自然言語処理 (NLP)、コンピュータビジョン (CV)、音声、さらには表形式のタスク向けに、最先端のモデルをノーコードでトレーニングできるツールです。

[W&B](https://wandb.com/) は Hugging Face AutoTrain と直接インテグレーションされており、実験管理と設定管理を提供します。実験のための CLI コマンドに 1 つのパラメータを使うだけで始められます。

{{< img src="/images/integrations/hf-autotrain-1.png" alt="実験メトリクスのログ" >}}

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

これらの変更を示すために、このページでは数学データセットで LLM を微調整し、[GSM8k Benchmarks](https://github.com/openai/grade-school-math) の `pass@1` で SOTA の結果を達成します。

## データセットを準備する

Hugging Face AutoTrain は、CSV のカスタム データセットが正しく動作するために所定のフォーマットになっていることを想定しています。

- トレーニング ファイルにはトレーニングで使用される `text` 列が必要です。最良の結果を得るには、`text` 列のデータを `### Human: Question?### Assistant: Answer.` というフォーマットに合わせてください。優れた例として [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) を参照してください。

    ただし、[MetaMathQA dataset](https://huggingface.co/datasets/meta-math/MetaMathQA) には `query`、`response`、`type` の列が含まれています。まずこのデータセットを前処理します。`type` 列を削除し、`query` と `response` 列の内容を結合して、新しい `text` 列に `### Human: Query?### Assistant: Response.` のフォーマットで格納してください。トレーニングでは、作成したデータセット [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath) を使用します。

## `autotrain` でトレーニングする

コマンドラインまたはノートブックから `autotrain` を使ってトレーニングを開始できます。`--log` 引数を使用するか、`--log wandb` を指定して、[W&B Run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) に結果をログできます。 

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
# ハイパーパラメーターを設定
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

# トレーニングを実行
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


{{< img src="/images/integrations/hf-autotrain-2.gif" alt="実験設定の保存" >}}

## 参考リソース

* [AutoTrain Advanced now supports Experiment Tracking](https://huggingface.co/blog/rishiraj/log-autotrain) by [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [Hugging Face AutoTrain Docs](https://huggingface.co/docs/autotrain/index)