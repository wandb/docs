---
description: The Prompts Quickstart shows how to visualise and debug the execution
  flow of your LLM chains and pipelines
displayed_sidebar: default
---

# 프롬프트 퀵스타트

[**여기에서 Colab 노트북으로 시도해보세요 →**](http://wandb.me/prompts-quickstart)

<head>
  <title>프롬프트 퀵스타트</title>
</head>

이 퀵스타트 가이드는 [Trace](intro.md)를 사용하여 LangChain, LlamaIndex 또는 자체 LLM 체인 또는 파이프라인에 대한 호출을 시각화하고 디버깅하는 방법을 안내합니다:

1. **[Langchain:](#use-wb-trace-with-langchain)** 자동 로그를 위해 LangChain 환경 변수 또는 컨텍스트 매니저 인테그레이션을 1줄로 사용하세요.

2. **[LlamaIndex:](#use-wb-trace-with-llamaindex)** LlamaIndex에서 W&B 콜백을 사용하여 자동 로깅합니다.

3. **[사용자 정의 사용](#use-wb-trace-with-any-llm-pipeline-or-plug-in)**: 자체 사용자 정의 체인 및 LLM 파이프라인 코드와 함께 Trace를 사용하세요.

## LangChain과 함께 W&B Trace 사용하기

:::info
**버전** `wandb >= 0.15.4` 및 `langchain >= 0.0.218`을 사용해 주세요
:::

LangChain으로부터 1줄 환경 변수를 사용하면, W&B Trace가 LangChain 모델, 체인 또는 에이전트에 대한 호출을 지속적으로 로깅하게 됩니다.

LangChain 문서에서 W&B Trace에 대한 문서도 확인할 수 있습니다. [LangChain 문서](https://python.langchain.com/docs/integrations/providers/wandb_tracing).

이 퀵스타트에서는 LangChain Math 에이전트를 사용할 것입니다:

### 1. LANGCHAIN_WANDB_TRACING 환경 변수 설정하기

먼저, LANGCHAIN_WANDB_TRACING 환경 변수를 true로 설정하세요. 이렇게 하면 LangChain과 함께 Weights & Biases 로깅이 자동으로 켜집니다:

```python
import os

# langchain에 대한 wandb 로깅 켜기
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
```

이제 LangChain LLM, 체인, 툴 또는 에이전트에 대한 모든 호출이 Weights & Biases에 로그됩니다.

### 2. Weights & Biases 설정 구성하기
선택적으로 Weights & Biases [환경 변수](/guides/track/environment-variables)를 설정하여 일반적으로 `wandb.init()`에 전달되는 파라미터를 설정할 수 있습니다. 일반적으로 사용되는 파라미터에는 로그가 전송되는 위치를 더 잘 제어하기 위해 `WANDB_PROJECT` 또는 `WANDB_ENTITY`가 포함됩니다. [`wandb.init`](../../ref/python/init.md)에 대한 자세한 내용은 API 참조 가이드를 참조하세요.

```python
# 선택적으로 wandb 설정 또는 구성을 설정하세요
os.environ["WANDB_PROJECT"] = "langchain-tracing"
```

### 3. LangChain 에이전트 생성하기
LangChain을 사용하여 표준 수학 에이전트를 생성하세요:

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 4. 에이전트를 실행하고 Weights & Biases 로깅 시작하기
LangChain을 평소와 같이 사용하여 에이전트를 호출하세요. Weights & Biases run이 시작되며 Weights & Biases **[API 키](https:wwww.wandb.ai/authorize)**를 입력하라는 메시지가 표시됩니다. API 키를 입력하면 에이전트 호출의 입력 및 출력이 Weights & Biases 앱으로 스트리밍되기 시작합니다.

```python
# 몇 가지 수학 문제 샘플
questions = [
    "5.4의 제곱근을 구하시오.",
    "3을 7.34로 나눈 후 파이의 거듭제곱을 하시오.",
    "0.47 라디안의 사인을 27의 세제곱근으로 나누시오.",
]

for question in questions:
    try:
        # 평소와 같이 에이전트를 호출하시오
        answer = math_agent.run(question)
        print(answer)
    except Exception as e:
        # 모든 오류도 Weights & Biases에 로그됩니다
        print(e)
        pass
```

각 에이전트 실행이 완료되면 LangChain 오브젝트의 모든 호출이 Weights & Biases에 로그됩니다

### 5. Weights & Biases에서 추적 보기

이전 단계에서 생성된 W&B [run](../runs/intro.md) 링크를 클릭하세요. 이렇게 하면 W&B 앱의 프로젝트 워크스페이스로 리디렉션됩니다.

생성한 run을 선택하여 추적 테이블, 추적 타임라인 및 LLM의 모델 아키텍처를 볼 수 있습니다.

![](/images/prompts/trace_timeline_detailed.png)

### 6. LangChain 컨텍스트 매니저
사용 사례에 따라 환경 변수를 해제하고 대신 컨텍스트 매니저를 사용하여 W&B에 로깅을 관리하는 것을 선호할 수 있습니다:

```python
from langchain.callbacks import wandb_tracing_enabled

# 환경 변수를 해제하고 대신 컨텍스트 매니저를 사용합니다
if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

# 컨텍스트 매니저를 사용하여 추적 활성화
with wandb_tracing_enabled():
    math_agent.run("5의 .123243 거듭제곱은 무엇인가요?")  # 이것은 추적되어야 합니다

math_agent.run("2의 .123243 거듭제곱은 무엇인가요?")  # 이것은 추적되지 않아야 합니다
```

이 LangChain 인테그레이션과 관련된 문제가 있으면 [wandb 저장소](https://github.com/wandb/wandb/issues)에 `langchain` 태그와 함께 문제를 보고해 주세요.

## LLM 파이프라인 또는 플러그인과 함께 W&B Trace 사용하기

:::info
**버전** `wandb >= 0.15.4`을 사용해 주세요
:::

W&B Trace는 하나 이상의 "span"을 로깅하여 생성됩니다. 기대되는 루트 span이 있으며, 이는 자체 자식 span을 받아들일 수 있고, 그 자식 span도 자신의 자식 span을 받아들일 수 있습니다. Span은 `AGENT`, `CHAIN`, `TOOL` 또는 `LLM` 유형일 수 있습니다.

Trace와 함께 로깅할 때, 단일 W&B run은 모델 또는 파이프라인에서 생성된 후 각 호출을 새로운 W&B run으로 시작할 필요 없이 Trace 테이블에 추가될 수 있습니다.

이 퀵스타트에서는 OpenAI 모델에 대한 단일 호출을 W&B Trace에 단일 span으로 로깅하는 방법을 보여줍니다. 그런 다음 더 복잡한 일련의 중첩된 span을 로깅하는 방법을 보여줍니다.

### 1. Trace를 가져오고 Weights & Biases run 시작하기

`wandb.init`을 호출하여 W&B run을 시작하세요. 여기에서 W&B 프로젝트 이름과 엔티티 이름(팀으로 로깅하는 경우)을 전달할 수 있으며, 구성 및 기타 항목도 포함됩니다. 모든 인수 목록은 [`wandb.init`](../../ref/python/init.md)을 참조하세요.

W&B run을 시작하면 Weights & Biases **[API 키](https:wwww.wandb.ai/authorize)**로 로그인하라는 메시지가 표시됩니다.


```python
import wandb

# 로깅할 wandb run 시작
wandb.init(project="trace-example")
```

W&B 팀으로 로깅하는 경우 `wandb.init`에서 `entity` 인수도 설정할 수 있습니다.

### 2. Trace에 로깅하기
이제 OpenAI 타임스에 쿼리를 하고 결과를 W&B Trace에 로깅할 것입니다. 입력 및 출력, 시작 및 종료 시간, OpenAI 호출이 성공했는지 여부, 토큰 사용량 및 추가 메타데이터를 로깅합니다.

Trace 클래스에 대한 인수의 전체 설명은 [여기](https://github.com/wandb/wandb/blob/653015a014281f45770aaf43627f64d9c4f04a32/wandb/sdk/data_types/trace_tree.py#L166)에서 볼 수 있습니다.

```python
import openai
import datetime
from wandb.sdk.data_types.trace_tree import Trace

openai.api_key = "<YOUR_OPENAI_API_KEY>"

# conifg를 정의하세요
model_name = "gpt-3.5-turbo"
temperature = 0.7
system_message = "언제나 마크다운을 사용해 3개의 간결한 항목으로 도움이 되는 답변을 하는 도우미입니다."

queries_ls = [
    "프랑스의 수도는 무엇인가요?",
    "달걀을 어떻게 삶나요?" * 10000,  # 고의로 openai 오류를 발생시킵니다
    "외계인이 도착하면 어떻게 하나요?",
]

for query in queries_ls:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    start_time_ms = datetime.datetime.now().timestamp() * 1000
    try:
        response = openai.ChatCompletion.create(
            model=model_name, messages=messages, temperature=temperature
        )

        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # 밀리세컨드 단위로 로깅됩니다
        status = "success"
        status_message = (None,)
        response_text = response["choices"][0]["message"]["content"]
        token_usage = response["usage"].to_dict()

    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # 밀리세컨드 단위로 로깅됩니다
        status = "error"
        status_message = str(e)
        response_text = ""
        token_usage = {}

    # wandb에 span을 생성합니다
    root_span = Trace(
        name="root_span",
        kind="llm",  # 종류는 "llm", "chain", "agent" 또는 "tool"일 수 있습니다
        status_code=status,
        status_message=status_message,
        metadata={
            "temperature": temperature,
            "token_usage": token_usage,
            "model_name": model_name,
        },
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        inputs={"system_prompt": system_message, "query": query},
        outputs={"response": response_text},
    )

    # span을 wandb에 로깅합니다
    root_span.log(name="openai_trace")
```

### 3. Weights & Biases에서 추적 보기

2단계에서 생성된 W&B [run](../runs/intro.md) 링크를 클릭하세요. 여기에서 LLM의 추적 테이블과 추적 타임라인을 볼 수 있어야 합니다.

### 4. 중첩된 span을 사용하여 LLM 파이프라인 로깅하기
이 예제에서는 에이전트가 호출되는 것을 시뮬레이션하고, 그 후에 LLM 체인이 호출되며, LLM 체인이 OpenAI LLM을 호출한 다음 에이전트가 계산기 툴을 "호출"합니다.

"에이전트" 실행의 각 단계에 대한 입력, 출력 및 메타데이터가 자체 span에 로깅됩니다. Span은 자식을 가질 수 있습니다.

```python
import time

# 에이전트가 답변해야 하는 쿼리
query = "다음 미국 선거까지 며칠입니까?"

# 1부 - 에이전트가 시작됩니다...
start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

root_span = Trace(
    name="MyAgent",
    kind="agent",
    start_time_ms=start_time_ms,
    metadata={"user": "optimus_12"},
)


# 2부 - 에이전트가 LLMChain을 호출합니다..
chain_span = Trace(name="LLMChain", kind="chain", start_time_ms=start_time_ms)

# 체인 span을 루트의 자식으로 추가합니다
root_span.add_child(chain_span)


# 3부 - LLMChain이 OpenAI LLM을 호출합니다...
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": query},
]

response = openai.ChatCompletion.create(
    model=model_name, messages=messages, temperature=temperature
)

llm_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
response_text = response["choices"][0]["message"]["content"]
token_usage = response["usage"].to_dict()

llm_span = Trace(
    name="OpenAI",
    kind="llm",
    status_code="success",
    metadata={
        "temperature": temperature,
        "token_usage": token_usage,
        "model_name": model_name,
    },
    start_time_ms=start_time_ms,
    end_time_ms=llm_end_time_ms,
    inputs={"system_prompt": system_message, "query": query},
    outputs={"response": response_text},
)

# LLM span을 체인 span의 자식으로 추가합니다...
chain_span.add_child(llm_span)

# 체인 span의 종료 시간을 업데이트합니다
chain_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"response": response_text}
)

# 체인 span의 종료 시간을 업데이트합니다
chain_span._span.end_time_ms = llm_end_time_ms


# 4부 - 에이전트가 툴을 호출합니다...
time.sleep(3)
days_to_election = 117
tool_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)

# 툴 span을 생성합니다
tool_span = Trace(
    name="Calculator",
    kind="tool",
    status_code="success",
    start_time_ms=llm_end_time_ms,
    end_time_ms=tool_end_time_ms,
    inputs={"input": response_text},
    outputs={"result": days_to_election},
)

# 툴 span을 루트의 자식으로 추가합니다
root_span.add_child(tool_span)


# 5부 - 툴에서 최종 결과가 추가됩니다
root_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"result": days_to_election}
)
root_span._span.end_time_ms = tool_end_time_ms


# 6부 - 루트 span을 로깅하여 W&B에 모든 span을 로깅합니다
root_span.log(name="openai_trace")
```

span을 로깅하면 W&B 앱에서 Trace 테이블이 업데이트되는 것을 볼 수 있습니다.

## LlamaIndex와 함께 W&B Trace 사용하기

:::info
**버전** `wandb >= 0.15.4` 및 `llama-index >= 0.6.35`를 사용해 주세요
:::

가장 낮은 수준에서, LlamaIndex는 로그를 추적하기 위해 시작/종료 이벤트([`CBEventTypes`](https://gpt-index.readthedocs.io/en/latest/reference/callbacks.html#llama_index.callbacks.CBEventType))의 개념을 사용합니다