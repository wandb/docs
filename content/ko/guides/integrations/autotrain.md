---
title: Hugging Face AutoTrain
menu:
  default:
    identifier: ko-guides-integrations-autotrain
    parent: integrations
weight: 130
---

[Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/index) 은 자연어 처리 (NLP) 작업, 컴퓨터 비전 (CV) 작업, 음성 작업 및 테이블 형식 작업에 대한 최첨단 models 트레이닝을 위한 노 코드 툴입니다.

[Weights & Biases](http://wandb.com/) 는 Hugging Face AutoTrain에 직접 통합되어 experiment 트래킹 및 config 관리를 제공합니다. 이는 experiments 를 위한 CLI 코맨드에서 단일 파라미터를 사용하는 것만큼 쉽습니다.

{{< img src="/images/integrations/hf-autotrain-1.png" alt="An example of logging the metrics of an experiment" >}}

## 필수 구성 요소 설치

`autotrain-advanced` 와 `wandb`를 설치합니다.

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

이러한 변경 사항을 보여주기 위해 이 페이지는 [GSM8k Benchmarks](https://github.com/openai/grade-school-math)에서 `pass@1`로 SoTA 결과를 얻기 위해 수학 데이터셋에서 LLM을 미세 조정합니다.

## 데이터셋 준비

Hugging Face AutoTrain은 제대로 작동하려면 CSV 커스텀 데이터셋이 특정 형식을 갖추어야 합니다.

- 트레이닝 파일에는 트레이닝에 사용되는 `text` 열이 있어야 합니다. 최상의 결과를 얻으려면 `text` 열의 데이터가 `### Human: Question?### Assistant: Answer.` 형식을 준수해야 합니다. [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)에서 훌륭한 예를 검토하십시오.

    그러나 [MetaMathQA 데이터셋](https://huggingface.co/datasets/meta-math/MetaMathQA)에는 `query`, `response` 및 `type` 열이 포함되어 있습니다. 먼저 이 데이터셋을 전처리합니다. `type` 열을 제거하고 `query` 및 `response` 열의 내용을 `### Human: Query?### Assistant: Response.` 형식으로 새 `text` 열에 결합합니다. 트레이닝은 결과 데이터셋인 [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath)를 사용합니다.

## `autotrain`을 사용하여 트레이닝

코맨드라인 또는 노트북에서 `autotrain` advanced를 사용하여 트레이닝을 시작할 수 있습니다. `--log` 인수를 사용하거나, `--log wandb`를 사용하여 [W&B run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에 결과를 기록합니다.

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
# 하이퍼파라미터 설정
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

# 트레이닝 실행
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


{{< img src="/images/integrations/hf-autotrain-2.gif" alt="An example of saving the configs of your experiment." >}}

## 추가 자료

* [AutoTrain Advanced now supports Experiment Tracking](https://huggingface.co/blog/rishiraj/log-autotrain) by [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [Hugging Face AutoTrain Docs](https://huggingface.co/docs/autotrain/index)
