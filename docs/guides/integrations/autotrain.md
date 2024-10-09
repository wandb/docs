---
title: Hugging Face AutoTrain
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[🤗 AutoTrain](https://huggingface.co/docs/autotrain/index)은 자연어 처리 (NLP) 작업, 컴퓨터 비전 (CV) 작업, 음성 작업 및 테이블 작업을 위한 최신 모델을 트레이닝하기 위한 코드가 필요 없는 툴입니다.

[Weights & Biases](http://wandb.com/)는 🤗 AutoTrain에 직접 통합되어 실험 추적 및 구성 관리 기능을 제공합니다. 실험을 위한 CLI 커맨드에 단일 파라미터를 사용하는 것만큼 간단합니다!

| ![실험의 메트릭이 로그되는 예시](/images/integrations/hf-autotrain-1.png) | 
|:--:| 
| **실험의 메트릭이 로그되는 예시.** |

## 시작하기

먼저 `autotrain-advanced`와 `wandb`를 설치해야 합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install --upgrade autotrain-advanced wandb
```

  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install --upgrade autotrain-advanced wandb
```

  </TabItem>
</Tabs>

## 시작하기: LLM 파인튜닝

이러한 변경 사항을 보여주기 위해 수학 데이터셋에서 LLM을 파인튜닝하여 [GSM8k Benchmarks](https://github.com/openai/grade-school-math)에서 `pass@1`의 최첨단 결과를 달성해 보겠습니다.

### 데이터셋 준비

🤗 AutoTrain은 당신의 CSV 커스텀 데이터셋이 특정 형식으로 되어 있어야 올바르게 작동합니다. 트레이닝 파일은 "text" 열을 포함해야 하며, 이 열에서 트레이닝이 진행됩니다. 최상의 결과를 위해 "text" 열은 `### Human: Question?### Assistant: Answer.` 형식으로 데이터가 있어야 합니다. AutoTrain Advanced가 기대하는 데이터셋의 훌륭한 예는 [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)입니다. 하지만 [MetaMathQA 데이터셋](https://huggingface.co/datasets/meta-math/MetaMathQA)을 살펴보면, "query", "response" 및 "type"의 세 가지 열이 있습니다. 우리는 "type" 열을 제거하고 "query"와 "response" 열의 내용을 `### Human: Query?### Assistant: Response.` 형식의 하나의 "text" 열로 결합하여 이 데이터셋을 사전 처리할 것입니다. 결과 데이터셋은 [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath)이며 트레이닝에 사용될 것입니다.

### Autotrain Advanced를 사용한 트레이닝

Autotrain Advanced CLI를 사용하여 트레이닝을 시작할 수 있습니다. 로그 기능을 활용하려면 간단히 `--log` 인수를 사용하세요. `--log wandb`를 지정하면 [W&B run](/guides/runs)에 결과가 자동으로 기록됩니다.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

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

  </TabItem>
  <TabItem value="notebook">

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

  </TabItem>
</Tabs>

| ![실험의 모든 구성 요소가 저장되는 예시.](/images/integrations/hf-autotrain-2.gif) | 
|:--:| 
| **실험의 모든 구성 요소가 저장되는 예시.** |

## 추가 자료

* [AutoTrain Advanced는 이제 Experiment Tracking을 지원합니다](https://huggingface.co/blog/rishiraj/log-autotrain) 작성자 [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [🤗 Autotrain 문서](https://huggingface.co/docs/autotrain/index)