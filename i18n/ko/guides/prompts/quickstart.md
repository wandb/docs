---
description: The Prompts Quickstart shows how to visualise and debug the execution
  flow of your LLM chains and pipelines
displayed_sidebar: default
---

# 프롬프트 퀵스타트

[**여기에서 Colab 노트북으로 시도해 보세요 →**](http://wandb.me/prompts-quickstart)

<head>
  <title>프롬프트 퀵스타트</title>
</head>

이 퀵스타트 가이드는 LangChain, LlamaIndex 또는 자신의 LLM 체인이나 파이프라인에 대한 호출을 시각화하고 디버깅하기 위해 [Trace](intro.md)를 사용하는 방법을 안내합니다:

1. **[Langchain:](#use-wb-trace-with-langchain)** 자동 로깅을 위한 LangChain 환경 변수 또는 컨텍스트 매니저 통합을 1줄로 사용하세요.

2. **[LlamaIndex:](#use-wb-trace-with-llamaindex)** LlamaIndex에서 Weights & Biases 콜백을 사용하여 자동 로깅하세요.

3. **[사용자 정의 사용](#use-wb-trace-with-any-llm-pipeline-or-plug-in)**: 자신만의 커스텀 체인 및 LLM 파이프라인 코드와 함께 Trace를 사용하세요.

## LangChain과 함께 W&B Trace 사용하기

:::info
**버전** `wandb >= 0.15.4` 및 `langchain >= 0.0.218`를 사용하세요
:::

LangChain의 1줄 환경 변수를 사용하면, W&B Trace가 LangChain 모델, 체인 또는 에이전트에 대한 호출을 지속적으로 로깅하게 됩니다.

W&B Trace에 대한 문서는 [LangChain 문서](https://python.langchain.com/docs/integrations/providers/wandb_tracing)에서도 볼 수 있습니다.

이 퀵스타트에서는 LangChain 수학 에이전트를 사용할 것입니다:

### 1. LANGCHAIN_WANDB_TRACING 환경 변수 설정하기

먼저, LANGCHAIN_WANDB_TRACING 환경 변수를 true로 설정하세요. 이렇게 하면 LangChain과 함께 자동 Weights & Biases 로깅이 활성화됩니다:

```python
import os

# langchain에 대한 wandb 로깅 켜기
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
```

그게 다입니다! 이제 LangChain LLM, 체인, 도구 또는 에이전트에 대한 모든 호출이 Weights & Biases에 로깅됩니다.

### 2. Weights & Biases 설정 구성하기
선택적으로, 일반적으로 `wandb.init()`에 전달되는 파라미터를 설정하기 위해 Weights & Biases [환경 변수](/guides/track/environment-variables)를 추가로 설정할 수 있습니다. 자주 사용되는 파라미터에는 로그가 전송되는 위치에 대한 더 많은 제어를 위해 `WANDB_PROJECT` 또는 `WANDB_ENTITY`가 포함됩니다. [`wandb.init`](../../ref/python/init.md)에 대한 자세한 정보는 API 참조 가이드를 참조하세요.

```python
# 선택적으로 wandb 설정 또는 구성 설정
os.environ["WANDB_PROJECT"] = "langchain-tracing"
```

### 3. LangChain 에이전트 만들기
LangChain을 사용하여 표준 수학 에이전트를 만드세요:

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 4. 에이전트 실행 및 Weights & Biases 로깅 시작하기
에이전트를 호출하듯이 LangChain을 정상적으로 사용하세요. Weights & Biases 실행이 시작되며 Weights & Biases **[API 키](https:wwww.wandb.ai/authorize)**를 요청받게 됩니다. API 키를 입력하면 에이전트 호출의 입력 및 출력이 Weights & Biases 앱으로 스트리밍되기 시작합니다.

```python
# 일부 샘플 수학 문제
questions = [
    "5.4의 제곱근을 찾으세요.",
    "3을 7.34로 나눈 후 파이의 거듭제곱을 하세요.",
    "0.47 라디안의 사인을 27의 세제곱근으로 나누세요.",
]

for question in questions:
    try:
        # 에이전트를 정상적으로 호출하세요
        answer = math_agent.run(question)
        print(answer)
    except Exception as e:
        # 모든 오류도 Weights & Biases에 로깅됩니다
        print(e)
        pass
```

각 에이전트 실행이 완료되면 LangChain 객체의 모든 호출이 Weights & Biases에 로깅됩니다

### 5. Weights & Biases에서 트레이스 보기

이전 단계에서 생성된 W&B [실행](../runs/intro.md) 링크를 클릭하세요. 이렇게 하면 W&B 앱에서 프로젝트 워크스페이스로 리디렉션됩니다.

생성한 실행을 선택하여 트레이스 테이블, 트레이스 타임라인 및 LLM의 모델 아키텍처를 볼 수 있습니다.

![](/images/prompts/trace_timeline_detailed.png)

### 6. LangChain 컨텍스트 매니저
사용 사례에 따라, 대신 컨텍스트 매니저를 사용하여 W&B에 대한 로깅을 관리하기를 선호할 수도 있습니다:

```python
from langchain.callbacks import wandb_tracing_enabled

# 환경 변수를 해제하고 대신 컨텍스트 매니저를 사용하세요
if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

# 컨텍스트 매니저를 사용하여 추적 활성화
with wandb_tracing_enabled():
    math_agent.run("5의 .123243 거듭제곱은 무엇인가요?")  # 이것은 추적되어야 합니다

math_agent.run("2의 .123243 거듭제곱은 무엇인가요?")  # 이것은 추적되지 않아야 합니다
```

이 LangChain 통합과 관련된 문제가 있으면 `langchain` 태그와 함께 [wandb 리포지토리](https://github.com/wandb/wandb/issues)에 보고해 주세요

## 어떤 LLM 파이프라인 또는 플러그인과 함께 W&B Trace 사용하기

:::info
**버전** `wandb >= 0.15.4`를 사용하세요
:::

W&B Trace는 하나 이상의 "스팬"을 로깅하여 생성됩니다. 루트 스팬이 예상되며, 중첩된 자식 스팬을 받을 수 있으며, 이 자식 스팬은 다시 자신의 자식 스팬을 받을 수 있습니다. 스팬은 `AGENT`, `CHAIN`, `TOOL` 또는 `LLM` 유형일 수 있습니다.

Trace로 로깅할 때 단일 W&B 실행은 모델 또는 파이프라인에서 생성된 각각의 호출에 대해 새로운 W&B 실행을 시작할 필요가 없으며, 대신 각 호출이 Trace 테이블에 추가됩니다.

이 퀵스타트에서는 단일 스팬으로 OpenAI 모델에 대한 단일 호출을 W&B Trace에 로깅하는 방법을 보여줄 것입니다. 그런 다음 중첩된 스팬의 더 복잡한 시리즈를 로깅하는 방법을 보여줄 것입니다.

### 1. Trace를 가져오고 Weights & Biases 실행을 시작하기

`wandb.init`을 호출하여 W&B 실행을 시작하세요. 여기서 W&B 프로젝트 이름과 W&B 팀에 로깅하는 경우 엔티티 이름을 전달할 수 있으며, 구성 및 기타 항목도 전달할 수 있습니다. 전체 인수 목록은 [`wandb.init`](../../ref/python/init.md)을 참조하세요.

Weights & Biases **[API 키](https:wwww.wandb.ai/authorize)**로 로그인하라는 메시지가 표시되면 W&B 실행을 시작하게 됩니다.

```python
import wandb

# 로깅을 위한 wandb 실행 시작
wandb.init(project="trace-example")
```

W&B 팀에 로깅하는 경우 `wandb.init`의 `entity` 인수도 설정할 수 있습니다.

### 2. Trace에 로깅하기
이제 OpenAI 타임즈를 쿼리하고 결과를 W&B Trace에 로깅할 것입니다. 입력 및 출력, 시작 및 종료 시간, OpenAI 호출이 성공했는지 여부, 토큰 사용량 및 추가 메타데이터를 로깅합니다.

Trace 클래스에 대한 인수의 전체 설명은 [여기](https://github.com/wandb/wandb/blob/653015a014281f45770aaf43627f64d9c4f04a32/wandb/sdk/data_types/trace_tree.py#L166)에서 볼 수 있습니다.

```python
import openai
import datetime
from wandb.sdk.data_types.trace_tree import Trace

openai.api_key = "<YOUR_OPENAI_API_KEY>"

# 구성을 정의하세요
model_name = "gpt-3.5-turbo"
temperature = 0.7
system_message = "당신은 항상 마크다운을 사용하여 3가지 간결한 항목으로 답변하는 유용한 조수입니다."

queries_ls = [
    "프랑스의 수도는 무엇인가요?",
    "계란을 어떻게 삶나요?" * 10000,  # 의도적으로 openai 오류를 트리거합니다
    "외계인이 도착하면 어떻게 해야 하나요?",
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
        )  # 밀리초 단위로 로깅됩니다
        status = "success"
        status_message = (None,)
        response_text = response["choices"][0]["message"]["content"]
        token_usage = response["usage"].to_dict()

    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # 밀리초 단위로 로깅됩니다
        status = "error"
        status_message = str(e)
        response_text = ""
        token_usage = {}

    # wandb에 스팬 만들기
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

    # span을 wandb에 로깅하기
    root_span.log(name="openai_trace")
```

### 3. Weights & Biases에서 트레이스 보기

2단계에서 생성된 W&B [실행](../runs/intro.md) 링크를 클릭하세요. 여기서 LLM의 트레이스 테이블과 트레이스 타임라인을 볼 수 있어야 합니다.

### 4. 중첩된 스팬을 사용하여 LLM 파이프라인 로깅하기
이 예제에서는 에이전트가 호출되고, 그 에이전트가 LLM 체인을 호출하고, LLM 체인이 OpenAI LLM을 호출하고, 그런 다음 에이전트가 계산기 도구를 "호출하는 것을 시뮬레이션할 것입니다.

"에이전트" 실행의 각 단계에 대한 입력, 출력 및 메타데이터가 각각의 스팬에 로깅됩니다. 스팬은 자식을 가질 수 있습니다

```python
import time

# 에이전트가 답변해야 하는 쿼리
query = "다음 미국 선거까지 며칠이 남았나요?"

# 1부 - 에이전트가 시작됩니다...
start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

root_span = Trace(
    name="MyAgent",
    kind="agent",
    start_time_ms=start_time_ms,
    metadata={"user": "optimus_12"},
)


# 2부 - 에이전트가 LLMChain을 호출합니다...
chain_span = Trace(name="LLMChain", kind="chain", start_time_ms=start_time_ms)

# 체인 스팬을 루트의 자식으로 추가합니다
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

# LLM 스팬을 체인 스팬의 자식으로 추가합니다...
chain_span.add_child(llm_span)

# 체인 스팬의 종료 시간을 업데이트합니다
chain_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"response": response_text}
)

# 체인 스팬의 종료 시간을 업데이트합니다
chain_span._span.end_time_ms = llm_end_time_ms


# 4부 - 에이전트가 도구를 호출합니다...
time.sleep(3)
days_to_election = 117
tool_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)

# 도구 스팬을 만듭니다
tool_span = Trace(
    name="Calculator",
    kind="tool",
    status_code="success",
    start_time_ms=llm_end_time_ms,
    end_time_ms=tool_end_time_ms,
    inputs={"input": response_text},
    outputs={"result": days_to_election},
)

# 도구 스팬을 루트의 자식으로 추가합니다
root_span.add_child(tool_span)


# 5부 - 도구에서 나온 최종 결과가 추가됩니다
root_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"result": days_to_election}
)
root_span._span.end_time_ms = tool_end_time_ms


# 6부 - 루트 스팬을 로깅하여 W&B에 모든 스팬을 로깅합니다
root_span.log(name="openai_trace")
```

스팬을 로깅한 후 W&B 앱에서 트레이스 테이블이 업데이트되는 것을 볼 수 있습니다.

## LlamaIndex와 함께 W

### 6. [선택사항] Weights & Biases 아티팩트에 인덱스 데이터 저장하기
Weights & Biases [아티팩트](guides/artifacts)는 버전 관리된 데이터 및 모델 저장소 제품입니다.

인덱스를 아티팩트에 로깅하고 필요할 때 사용함으로써, 특정 호출에서 인덱스에 어떤 데이터가 있었는지에 대한 전체 가시성을 보장하며 로깅된 Trace 출력물과 특정 버전의 인덱스를 연결할 수 있습니다.

```python
# `index_name`에 전달된 문자열이 아티팩트 이름이 됩니다
wandb_callback.persist_index(index, index_name="my_vector_store")
```

그런 다음 W&B 실행 페이지의 아티팩트 탭으로 이동하여 업로드된 인덱스를 볼 수 있습니다.

**W&B 아티팩트에 저장된 인덱스 사용하기**

아티팩트에서 인덱스를 로드하면 [`StorageContext`](https://gpt-index.readthedocs.io/en/latest/reference/storage.html)를 반환합니다. 이 저장소 컨텍스트를 사용하여 LlamaIndex [로딩 함수](https://gpt-index.readthedocs.io/en/latest/reference/storage/indices_save_load.html) 중 하나를 사용하여 메모리에 인덱스를 로드합니다.

```python
from llama_index import load_index_from_storage

storage_context = wandb_callback.load_storage_context(
    artifact_url="<entity/project/index_name:version>"
)
index = load_index_from_storage(storage_context, service_context=service_context)
```

**참고:** [`ComposableGraph`](https://gpt-index.readthedocs.io/en/latest/reference/query/query_engines/graph_query_engine.html)의 경우 인덱스의 루트 ID는 W&B 앱의 아티팩트 메타데이터 탭에서 찾을 수 있습니다.

## 다음 단계

- 기존 W&B 기능인 테이블과 실행을 사용하여 LLM 애플리케이션 성능을 추적할 수 있습니다. 더 알아보려면 이 튜토리얼을 참조하세요:
[튜토리얼: LLM 애플리케이션 성능 평가](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)