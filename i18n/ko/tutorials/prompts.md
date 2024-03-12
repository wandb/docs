
# LLM에 대한 반복 작업

[**여기에서 Colab 노트북에서 시도해 보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb)

**Weights & Biases Prompts**는 LLM 기반 애플리케이션 개발을 위해 구축된 LLMOps 툴 모음입니다.

W&B Prompts를 사용하여 LLM의 실행 흐름을 시각화하고 검사하고, LLM의 입력과 출력을 분석하고, 중간 결과를 보고, 프롬프트와 LLM 체인 설정을 안전하게 저장하고 관리하세요.

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
print("OpenAI API 키가 설정되었습니다")
```

# W&B Prompts

W&B는 현재 __Trace__라고 불리는 툴을 지원합니다. Trace는 세 가지 주요 구성 요소로 구성됩니다:

**Trace 테이블**: 체인의 입력과 출력에 대한 개요입니다.

**Trace 타임라인**: 체인의 실행 흐름을 보여주며 구성 요소 유형에 따라 색상이 다르게 표시됩니다.

**모델 아키텍처**: 체인의 구조와 각 체인 구성 요소를 초기화하는 데 사용된 파라미터에 대한 세부 정보를 확인할 수 있습니다.

이 섹션을 실행한 후, 워크스페이스에 자동으로 생성된 새 패널을 볼 수 있으며, 각 실행, 트레이스 및 모델 아키텍처를 표시합니다.


![prompts_1](/images/tutorials/prompts_quickstart/prompts.png)

![prompts_2](/images/tutorials/prompts_quickstart/prompts2.png)


`WandbTracer`을 가져오고, 나중에 `WandbTracer`에 전달될 `wandb.init()`에 대한 인수 사전을 선택적으로 정의하세요. 이에는 프로젝트 이름, 팀 이름, 엔티티 등이 포함됩니다. wandb.init에 대한 자세한 내용은 API 참조 가이드를 참조하세요.


```python
from wandb.integration.langchain import WandbTracer

wandb_config = {"project": "wandb_prompts_quickstart"}
```

### LangChain을 사용한 수학


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

LangChain 체인이나 에이전트를 호출할 때 `WandbTracer`를 전달하여 W&B에 트레이스를 로그하세요.


```python
questions = [
    "5.4의 제곱근을 찾으세요.",
    "3을 7.34로 나눈 값에 파이의 거듭제곱을 한 값은 무엇인가요?",
    "0.47 라디안의 사인 값을 27의 세제곱근으로 나눈 값은 무엇인가요?",
    "1을 0으로 나눈 값은 무엇인가요?"
]
for question in questions:
  try:
    answer = agent.run(question, callbacks=[WandbTracer(wandb_config)])
    print(answer)
  except Exception as e:
    print(e)
    pass
```

세션을 마친 후, `WandbTracer.finish()`를 호출하여 wandb run이 깨끗하게 종료되도록 하는 것이 최선의 방법입니다.


```python
WandbTracer.finish()
```

# Lang Chain 구현이 아닌 경우

Langchain을 사용하지 않고 싶은 경우, 특히 팀의 코드에 인테그레이션을 작성하거나 구현하고 싶은 경우가 있습니다. 전혀 문제없습니다! `TraceTree`와 `Span`에 대해 알아봅시다!

![prompts_3](/images/tutorials/prompts_quickstart/prompts3.png)

**참고:** W&B Runs는 한 번의 실행에 필요한 만큼 많은 트레이스를 로깅할 수 있으므로, 새로운 실행을 매번 생성할 필요 없이 `run.log`의 여러 번 호출이 가능합니다.


```python
from wandb.sdk.data_types import trace_tree
import wandb
```

Span은 작업 단위를 나타내며, Span은 `AGENT`, `TOOL`, `LLM` 또는 `CHAIN` 유형을 가질 수 있습니다


```python
parent_span = trace_tree.Span(
  name="Example Span", 
  span_kind = trace_tree.SpanKind.AGEN
)
```

Span은 중첩될 수 있습니다 (그리고 중첩되어야 합니다!):


```python
# 툴 호출에 대한 스팬 생성
tool_span = trace_tree.Span(
  name="Tool 1", 
  span_kind = trace_tree.SpanKind.TOOL
)

# LLM 체인 호출에 대한 스팬 생성
chain_span = trace_tree.Span(
  name="LLM CHAIN 1", 
  span_kind = trace_tree.SpanKind.CHAIN
)

# LLM 체인에 의해 호출된 LLM에 대한 호출 스팬 생성
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

아래와 같이 W&B에 parent_span을 로그할 수 있습니다.


```python
run = wandb.init(name="manual_span_demo", project="wandb_prompts_demo")
run.log({"trace": trace_tree.WBTraceTree(parent_span)})
run.finish()
```

생성된 W&B Run 링크를 클릭하면 생성된 Trace를 검사할 수 있는 워크스페이스로 이동합니다.