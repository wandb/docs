---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 허깅 페이스 오토트레인

[🤗 오토트레인](https://huggingface.co/docs/autotrain/index)은 자연어 처리(NLP) 작업, 컴퓨터 비전(CV) 작업, 음성 작업 및 표 형식 작업을 위한 최신 모델을 학습하기 위한 노코드 도구입니다.

[Weights & Biases](http://wandb.com/)는 🤗 오토트레인에 직접 통합되어, 실험 추적 및 구성 관리를 제공합니다. 실험을 위한 CLI 명령에서 단일 인수를 사용하는 것만큼 쉽습니다!

| ![실험의 메트릭이 어떻게 기록되는지에 대한 예시](@site/static/images/integrations/hf-autotrain-1.png) | 
|:--:| 
| **실험의 메트릭이 어떻게 기록되는지에 대한 예시입니다.** |

## 시작하기

먼저, `autotrain-advanced`와 `wandb`를 설치해야 합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: '명령 줄', value: 'script'},
    {label: '노트북', value: 'notebook'},
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

## 시작하기: LLM 파인 튜닝

이 변경 사항을 보여주기 위해, 수학 데이터세트에서 LLM에 대해 파인 튜닝을 하고 [GSM8k 벤치마크](https://github.com/openai/grade-school-math)에서 `pass@1`에서 SoTA 결과를 달성하려고 합니다.

### 데이터세트 준비하기

🤗 오토트레인은 학습을 수행할 "text" 열이 포함된 특정 형식의 CSV 사용자 정의 데이터세트를 기대합니다. 최상의 결과를 얻으려면 "text" 열에는 `### Human: 질문?### Assistant: 답변.` 형식의 데이터가 있어야 합니다. 오토트레인 고급이 기대하는 데이터세트 유형의 훌륭한 예시는 [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)입니다. 그러나 [MetaMathQA 데이터세트](https://huggingface.co/datasets/meta-math/MetaMathQA)를 살펴보면, "query", "response" 및 "type"의 3개 열이 있습니다. 이 데이터세트를 전처리하여 "type" 열을 제거하고 "query" 및 "response" 열의 내용을 `### Human: 질문?### Assistant: 응답.` 형식으로 하나의 "text" 열 아래에 결합할 것입니다. 결과 데이터세트는 [`rishiraj/guanaco-style-metamath`](https://huggingface.co/datasets/rishiraj/guanaco-style-metamath)이며 학습에 사용될 것입니다.

### 오토트레인 고급을 사용한 학습

오토트레인 고급 CLI를 사용하여 학습을 시작할 수 있습니다. 로깅 기능을 활용하려면 `--log` 인수를 단순히 사용하면 됩니다. `--log wandb`를 지정하면 결과가 [W&B 실행](https://docs.wandb.ai/guides/runs)에 자연스럽게 기록됩니다.

<Tabs
  defaultValue="script"
  values={[
    {label: '명령 줄', value: 'script'},
    {label: '노트북', value: 'notebook'},
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

# 학습 실행
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

| ![실험의 모든 구성이 어떻게 저장되는지에 대한 예시.](@site/static/images/integrations/hf-autotrain-2.gif) | 
|:--:| 
| **실험의 모든 구성이 어떻게 저장되는지에 대한 예시입니다.** |

## 추가 자료

* [AutoTrain Advanced 이제 실험 추적을 지원합니다](https://huggingface.co/blog/rishiraj/log-autotrain) by [Rishiraj Acharya](https://huggingface.co/rishiraj).
* [🤗 오토트레인 문서](https://huggingface.co/docs/autotrain/index)