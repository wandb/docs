---
title: Hugging Face AutoTrain
menu:
  default:
    identifier: ko-guides-integrations-autotrain
    parent: integrations
weight: 130
---

[Hugging Face AutoTrain](https://huggingface.co/docs/autotrain/index)은 코드 작성 없이 최첨단 자연어 처리(NLP), 컴퓨터 비전(CV), 음성, 그리고 테이블 형태 데이터 작업을 위한 모델을 트레이닝할 수 있는 툴입니다.

[W&B](https://wandb.com/)는 Hugging Face AutoTrain에 직접 인테그레이션되어 있어, 실험 추적과 설정 관리가 가능합니다. 실험에 CLI 커맨드의 한 파라미터만 추가해서 아주 간편하게 사용할 수 있습니다.

{{< img src="/images/integrations/hf-autotrain-1.png" alt="Experiment metrics logging" >}}

## 필수 패키지 설치

`autotrain-advanced`와 `wandb`를 설치합니다.

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

이 가이드에서는 LLM을 수학 데이터셋에 파인튜닝하여 [GSM8k Benchmarks](https://github.com/openai/grade-school-math)의 `pass@1`에서 SoTA 결과를 달성하는 예시를 다룹니다.

## 데이터셋 준비

Hugging Face AutoTrain은 CSV 커스텀 데이터셋이 올바른 포맷을 가져야 제대로 작동합니다.

- 트레이닝 파일에는 반드시 `text` 컬럼이 있어야 하며, 트레이닝에 사용됩니다. 최상의 결과를 원한다면, `text` 컬럼 데이터는 `### Human: Question?### Assistant: Answer.` 형태를 따라야 합니다. [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)에서 좋은 예시를 확인해보세요.

    하지만, [MetaMathQA 데이터셋](https://huggingface.co/datasets/meta-math/MetaMathQA)에는 `query`, `response`, `type` 컬럼이 있습니다. 먼저 이 데이터셋을 전처리해야 합니다. `type` 컬럼은 제거하고, `query`와 `response` 컬럼의 내용을 합쳐서 `### Human: Query?### Assistant: Response.` 형태의 새로운 `text` 컬럼을 만들어주세요. 이 과정을 통해 만들어진 데이터셋은 [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath)입니다.

## `autotrain`으로 트레이닝 실행하기

커맨드라인이나 노트북에서 `autotrain` advanced로 트레이닝을 바로 시작할 수 있습니다. `--log` 인수를 사용하거나, `--log wandb`로 [W&B Run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에 결과를 기록할 수 있습니다.

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


{{< img src="/images/integrations/hf-autotrain-2.gif" alt="Experiment config saving" >}}

## 추가 자료

* [AutoTrain Advanced now supports Experiment Tracking](https://huggingface.co/blog/rishiraj/log-autotrain) by [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [Hugging Face AutoTrain Docs](https://huggingface.co/docs/autotrain/index)