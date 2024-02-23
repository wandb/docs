
# LLM에 대한 반복 작업

[**Colab 노트북에서 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb)

**Weights & Biases Prompts**는 LLM 기반 애플리케이션 개발을 위해 구축된 LLMOps 도구 모음입니다.

W&B Prompts를 사용하여 LLM의 실행 흐름을 시각화하고 검사하고, LLM의 입력 및 출력을 분석하고, 중간 결과를 보고, 프롬프트와 LLM 체인 구성을 안전하게 저장 및 관리하세요.

## 설치


```python
!pip install "wandb==0.15.2" -qqq
!pip install "langchain==v0.0.158" openai -qqq
```

## 설정

이 데모는 [OpenAI 키](https://platform.openai.com)가 있어야 합니다.


```python
import os
from getpass import getpass

if os.getenv("OPENAI_API_KEY") is None:
  os.environ["OPENAI_API_KEY"] = getpass("https://platform.openai.com/account/api-keys에서 OpenAI 키를 붙여넣으세요\n")
assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "유효한 OpenAI API 키로 보이지 않습니다"
print("OpenAI API 키가 구성되었습니다")
```

# W&B Prompts

W&B는 현재 __Trace__라는 도구를 지원합니다. Trace는 세 가지 주요 구성 요소로 구성됩니다:

**Trace 테이블**: 체인의 입력 및 출력 개요입니다.

**Trace 타임라인**: 체인의 실행 흐름을 표시하며 구성 요소 유형에 따라 색상으로 구분됩니다.

**모델 아키텍처**: 체인의 구조 및 체인의 각 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 봅니다.

이 섹션을 실행한 후에는 워크스페이스에 자동으로 생성된 새 패널에서 각 실행, 추적 및 모델 아키텍처를 볼 수 있습니다


![prompts_1](/images/tutorials/prompts_quickstart/prompts.png)

![prompts_2](/images/tutorials/prompts_quickstart/prompts2.png)


`WandbTracer`를 가져와서 나중에 `WandbTracer`에 전달될 `wandb.init()`에 대한 인수 사전을 선택적으로 정의하세요. 여기에는 프로젝트 이름, 팀 이름, 엔티티 등이 포함됩니다. wandb.init에 대한 자세한 정보는 API 참조 가이드를 참조하세요.


```python
from wandb.integration.langchain import WandbTracer

wandb_config = {"project": "wandb_prompts_quickstart"}
```

### LangChain으로 수학하기


```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
```


```python
llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
  tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

LangChain 체인이나 에이전트를 호출할 때 `WandbTracer`을 전달하여 W&B에 추적을 기록하세요


```python
questions = [
    "5.4의 제곱근을 찾으세요.",
    "3을 7.34로 나눈 값에 파이의 거듭제곱을 하세요.",
    "0.47 라디안의 사인을 27의 세제곱근으로 나눈 값은 무엇입니까?",
    "1을 0으로 나눈 값은 무엇입니까"
]
for question in questions:
  try:
    answer = agent.run(question, callbacks=[WandbTracer(wandb_config)])
    print(answer)
  except Exception as e:
    print(e)
    pass
```

세션을 마친 후에는 `WandbTracer.finish()`를 호출하여 wandb 실행이 깔끔하게 종료되도록 하는 것이 최선의 방법입니다.


```python
WandbTracer.finish()
```

# Lang Chain이 아닌 구현

Langchain을 사용하지 않으려는 경우, 특히 통합을 작성하거나 팀의 코드를 작성하고 싶은 경우는 어떻게 될까요? 완전히 괜찮습니다! `TraceTree` 및 `Span`에 대해 알아봅시다!

![prompts_3](/images/tutorials/prompts_quickstart/prompts3.png)

**참고:** W&B 실행은 단일 실행에 필요한 만큼 많은 추적을 기록할 수 있으므로, 새로운 실행을 매번 생성할 필요 없이 여러 번의 `run.log` 호출이 가능합니다


```python
from wandb.sdk.data_types import trace_tree
import wandb
```

Span은 작업의 단위를 나타내며, Span의 유형은 `AGENT`, `TOOL`, `LLM` 또는 `CHAIN`일 수 있습니다


```python
parent_span = trace_tree.Span(
  name="Example Span", 
  span_kind = trace_tree.SpanKind.AGEN
)
```

Span은 중첩될 수 있습니다 (그리고 그래야 합니다!):


```python
# 도구 호출에 대한 span 생성
tool_span = trace_tree.Span(
  name="Tool 1", 
  span_kind = trace_tree.SpanKind.TOOL
)

# LLM 체인 호출에 대한 span 생성
chain_span = trace_tree.Span(
  name="LLM CHAIN 1", 
  span_kind = trace_tree.SpanKind.CHAIN
)

# LLM 체인에 의해 호출된 LLM에 대한 span 생성
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

아래와 같이 parent_span을 W&B에 로그할 수 있습니다.


```python
run = wandb.init(name="manual_span_demo", project="wandb_prompts_demo")
run.log({"trace": trace_tree.WBTraceTree(parent_span)})
run.finish()
```

생성된 추적을 검사할 수 있는 워크스페이스로 이동하는 W&B 실행 링크를 클릭하면 됩니다.