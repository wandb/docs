---
title: 사용 예시
description: 실제 코드 예제를 통해 W&B Inference 사용 방법을 알아보세요.
linkTitle: Examples
menu:
  default:
    identifier: ko-guides-inference-examples
weight: 50
---

이 예제들은 W&B Inference와 Weave를 활용하여 트레이싱, 평가, 그리고 비교하는 방법을 보여줍니다.

## 기본 예제: Weave로 Llama 3.1 8B 트레이싱하기

이 예제에서는 **Llama 3.1 8B** 모델에 프롬프트를 보내고 Weave에서 호출을 트레이스하는 방법을 다룹니다. 트레이싱은 LLM 호출의 전체 입력과 출력을 캡처하고, 성능을 모니터링하며, Weave UI에서 결과를 분석할 수 있도록 해줍니다.

{{< alert title="팁" >}}
[Weave에서 트레이싱하는 법](https://weave-docs.wandb.ai/guides/tracking/tracing)에 대해 더 알아보세요.
{{< /alert >}}

이 예제에서 다루는 내용:
- `@weave.op()` 데코레이터가 적용된 함수에서 채팅 완성 요청을 보냅니다.
- 트레이스가 여러분의 W&B entity와 project에 기록되고 연결됩니다.
- 함수는 자동으로 트레이스되어 입력값, 출력값, 지연 시간, 메타데이터를 로그로 남깁니다.
- 결과는 터미널에 출력되고, 트레이스는 [https://wandb.ai](https://wandb.ai) 의 **Traces** 탭에서 확인할 수 있습니다.

이 예제를 실행하기 전에, [사전 준비 사항]({{< relref path="prerequisites" lang="ko" >}}) 을 완료해 주세요.

```python
import weave
import openai

# 트레이싱을 위한 Weave 팀과 프로젝트를 설정합니다
weave.init("<your-team>/<your-project>")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',

    # https://wandb.ai/authorize에서 API 키를 확인하세요
    api_key="<your-api-key>",
)

# 모델 호출을 Weave에서 트레이스합니다
@weave.op()
def run_chat():
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."}
        ],
    )
    return response.choices[0].message.content

# 트레이스된 호출을 실행하고 로그를 남깁니다
output = run_chat()
print(output)
```

코드를 실행한 후, Weave에서 트레이스를 확인하는 방법은 다음과 같습니다:
1. 터미널에 출력된 링크를 클릭합니다 (예: `https://wandb.ai/<your-team>/<your-project>/r/call/01977f8f-839d-7dda-b0c2-27292ef0e04g`)
2. 또는 [https://wandb.ai](https://wandb.ai) 로 이동하여 **Traces** 탭을 선택합니다.

## 고급 예제: Weave 평가 및 리더보드 활용하기

모델 호출을 트레이싱하는 것 외에도, 성능 평가 및 리더보드 발행도 할 수 있습니다.  
이 예제에서는 두 모델을 질문-답변 데이터셋에서 비교하며, 클라이언트 초기화 단계에서 로그를 보낼 project 이름을 지정합니다.

이 예제를 실행하기 전에, [사전 준비 사항]({{< relref path="prerequisites" lang="ko" >}}) 을 완료해 주세요.

```python
import os
import asyncio
import openai
import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

# 트레이싱을 위한 Weave 팀과 프로젝트를 설정합니다
weave.init("<your-team>/<your-project>")

dataset = [
    {"input": "What is 2 + 2?", "target": "4"},
    {"input": "Name a primary color.", "target": "red"},
]

@weave.op
def exact_match(target: str, output: str) -> float:
    return float(target.strip().lower() == output.strip().lower())

class WBInferenceModel(weave.Model):
    model: str

    @weave.op
    def predict(self, prompt: str) -> str:
        client = openai.OpenAI(
            base_url="https://api.inference.wandb.ai/v1",
            # https://wandb.ai/authorize에서 API 키를 확인하세요
            api_key="<your-api-key>",
            # 선택 사항: 로그 저장 위치를 지정합니다
            project="<your-team>/<your-project>"
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

llama = WBInferenceModel(model="meta-llama/Llama-3.1-8B-Instruct")
deepseek = WBInferenceModel(model="deepseek-ai/DeepSeek-V3-0324")

def preprocess_model_input(example):
    return {"prompt": example["input"]}

evaluation = weave.Evaluation(
    name="QA",
    dataset=dataset,
    scorers=[exact_match],
    preprocess_model_input=preprocess_model_input,
)

async def run_eval():
    await evaluation.evaluate(llama)
    await evaluation.evaluate(deepseek)

asyncio.run(run_eval())

spec = leaderboard.Leaderboard(
    name="Inference Leaderboard",
    description="Compare models on a QA dataset",
    columns=[
        leaderboard.LeaderboardColumn(
            evaluation_object_ref=get_ref(evaluation).uri(),
            scorer_name="exact_match",
            summary_metric_path="mean",
        )
    ],
)

weave.publish(spec)
```

코드를 모두 실행한 후, [https://wandb.ai/](https://wandb.ai/) 에서 다음을 확인할 수 있습니다:

- **Traces** 탭을 선택해 [트레이스를 확인](https://weave-docs.wandb.ai/guides/tracking/tracing)하세요
- **Evals** 탭을 선택해 [모델 평가 결과를 확인](https://weave-docs.wandb.ai/guides/core-types/evaluations)하세요
- **Leaders** 탭을 선택해 [생성된 리더보드 보기](https://weave-docs.wandb.ai/guides/core-types/leaderboards)  

{{< img src="/images/inference/inference-advanced-evals.png" alt="모델 평가 결과 보기" >}}

{{< img src="/images/inference/inference-advanced-leaderboard.png" alt="리더보드 보기" >}}

## 다음 단계

- 사용 가능한 모든 메소드는 [API reference]({{< relref path="api-reference" lang="ko" >}})에서 확인해 보세요
- [UI]({{< relref path="ui-guide" lang="ko" >}})에서 다양한 모델을 직접 시도해 보세요