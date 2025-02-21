---
title: Hugging Face AutoTrain
menu:
  default:
    identifier: ja-guides-integrations-autotrain
    parent: integrations
weight: 130
---

[Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/index) は、自然言語処理 (NLP) タスク、コンピュータビジョン (CV) タスク、音声タスク、そして表形式タスクのための最先端モデルをトレーニングするためのコード不要のツールです。

[Weights & Biases](http://wandb.com/) は Hugging Face AutoTrain に直接統合されており、実験管理と設定管理を提供します。これは、CLI コマンド内で実験のための単一のパラメータを使用するのと同じくらい簡単です。

{{< img src="/images/integrations/hf-autotrain-1.png" alt="実験のメトリクスをログする例" >}}

## 事前準備のインストール

`autotrain-advanced` と `wandb` をインストールします。

{{< tabpane text=true >}}

{{% tab header="Command Line" value="script" %}}

```shell
pip install --upgrade autotrain-advanced wandb
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

```notebook
!pip install --upgrade autotrain-advanced wandb
```

{{% /tab %}}

{{< /tabpane >}}

これらの変更をデモするために、このページでは数学のデータセットで LLM を微調整し、[GSM8k ベンチマーク](https://github.com/openai/grade-school-math) の `pass@1` で最先端の結果を達成します。

## データセットの準備

Hugging Face AutoTrain は、あなたの CSV カスタムデータセットが正しく機能するために特定の形式を持つことを期待しています。

- あなたのトレーニングファイルには、`text` 列が含まれている必要があります。トレーニングはそれを使用します。最良の結果を得るためには、`text` 列のデータは `### Human: Question?### Assistant: Answer.` 形式に準拠している必要があります。素晴らしい例として、[`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) を確認してください。

    しかし、[MetaMathQA データセット](https://huggingface.co/datasets/meta-math/MetaMathQA) には、`query`、`response`、および `type` の列が含まれています。まず、このデータセットを前処理します。`type` 列を削除し、`query` と `response` 列の内容を新しい `text` 列として `### Human: Query?### Assistant: Response.` 形式で結合します。トレーニングには、結果として得られるデータセット、[`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath) を使用します。

## `autotrain` を使用したトレーニング

コマンドラインまたはノートブックから `autotrain` 高度を使用してトレーニングを開始できます。`--log` 引数を使用するか、結果を [W&B run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) にログするために `--log wandb` を使用します。

{{< tabpane text=true >}}

{{% tab header="Command Line" value="script" %}}

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

{{% tab header="Notebook" value="notebook" %}}

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

{{< img src="/images/integrations/hf-autotrain-2.gif" alt="実験の設定を保存する例" >}}

## 他のリソース

* [AutoTrain Advanced による実験管理のサポート](https://huggingface.co/blog/rishiraj/log-autotrain) -- [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [Hugging Face AutoTrain ドキュメント](https://huggingface.co/docs/autotrain/index)