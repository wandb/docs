---
title: Iterate on LLMs
displayed_sidebar: tutorials
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb"></CTAButtons>

**Weights & Biases Prompts**는 LLM 기반 애플리케이션 개발을 위한 LLMOps 툴 모음입니다.

W&B Prompts를 사용하여 LLM의 실행 흐름을 시각화하고 검사하며, LLM의 입력 및 출력을 분석하고, 중간 결과를 확인하며, 프롬프트와 LLM 체인 설정을 안전하게 저장하고 관리하십시오.

## 설치

```python
!pip install "wandb==0.15.2" -qqq
```
# W&B Prompts

W&B는 현재 __Trace__라는 툴을 지원합니다. Trace는 세 가지 주요 구성 요소로 이루어져 있습니다:

**Trace table**: 체인의 입력 및 출력 개요.

**Trace timeline**: 체인의 실행 흐름을 표시하며 구성 요소 유형에 따라 색상이 지정됩니다.

**Model architecture**: 체인의 구조 및 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 봅니다.

다음 이미지에서는 워크스페이스에 자동으로 생성된 새 패널이 각 실행, 추적, 그리고 모델 아키텍처를 보여줍니다.

![prompts_1](/images/tutorials/prompts_quickstart/prompts.png)

![prompts_2](/images/tutorials/prompts_quickstart/prompts2.png)

# 사용자 지정 인테그레이션 작성하기

인테그레이션을 작성하거나 자신의 코드를 계측하고 싶다면 어떻게 해야 할까요? 이때 `TraceTree`와 `Span`과 같은 유틸리티가 유용합니다.

![prompts_3](/images/tutorials/prompts_quickstart/prompts3.png)

**참고:** W&B Runs는 필요한 만큼 많은 트레이스를 단일 run에 기록하는 것을 지원합니다. 즉, 매번 새롭게 Run을 생성할 필요 없이 `run.log`를 여러 번 호출할 수 있습니다.

```python
from wandb.sdk.data_types import trace_tree
import wandb
```

Span은 작업 단위를 나타내며, Spans는 `AGENT`, `TOOL`, `LLM` 또는 `CHAIN` 유형을 가질 수 있습니다.

```python
parent_span = trace_tree.Span(
  name="Example Span", 
  span_kind = trace_tree.SpanKind.AGEN
)
```

Span은 중첩될 수 있습니다 (그리고 중첩해야 합니다!).

```python
# 툴 호출을 위한 span 생성
tool_span = trace_tree.Span(
  name="Tool 1", 
  span_kind = trace_tree.SpanKind.TOOL
)

# LLM 체인 호출을 위한 span 생성
chain_span = trace_tree.Span(
  name="LLM CHAIN 1", 
  span_kind = trace_tree.SpanKind.CHAIN
)

# LLM 체인에 의해 호출되는 LLM을 위한 span 생성
llm_span = trace_tree.Span(
  name="LLM 1", 
  span_kind = trace_tree.SpanKind.LLM
)
chain_span.add_child_span(llm_span)
```

Span 입력 및 출력은 다음과 같이 추가할 수 있습니다:

```python
tool_span.add_named_result(
  {"input": "search: google founded in year"}, 
  {"response": "1998"}
)
chain_span.add_named_result(
  {"input": "calculate: 2023 - 1998"}, 
  {"response": "25"}
)
llm_span.add_named_result(
  {"input": "calculate: 2023 - 1998", "system": "you are a helpful assistant", }, 
  {"response": "25", "tokens_used":218}
)

parent_span.add_child_span(tool_span)
parent_span.add_child_span(chain_span)

parent_span.add_named_result({"user": "calculate: 2023 - 1998"}, 
                             {"response": "25 years old"})
```

그런 다음 위의 방식대로 parent_span을 W&B에 로그할 수 있습니다. 

```python
run = wandb.init(name="manual_span_demo", project="wandb_prompts_demo")
run.log({"trace": trace_tree.WBTraceTree(parent_span)})
run.finish()
```

생성된 W&B Run 링크를 클릭하면 생성된 Trace를 검사할 수 있는 워크스페이스로 이동합니다.