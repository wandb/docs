---
title: Tutorial: Visualize and debug LLMs with Prompts
description: Prompts 퀵스타트에서는 LLM 체인과 파이프라인의 실행 흐름을 시각화하고 디버그하는 방법을 보여줍니다
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb"></CTAButtons>

이 퀵스타트 가이드는 LangChain, LlamaIndex 또는 사용자 정의 LLM Chain이나 파이프라인에 호출을 시각화하고 디버그하는 방법을 안내합니다:

1. **[Langchain:](#use-wb-trace-with-langchain)** 자동 로그를 위한 1줄 LangChain 환경 변수 또는 컨텍스트 관리자를 사용한 인테그레이션.

2. **[LlamaIndex:](#use-wb-trace-with-llamaindex)** 자동 로그를 위한 LlamaIndex의 W&B 콜백 사용.

3. **[Custom usage](#use-wb-trace-with-any-llm-pipeline-or-plug-in)**: 사용자 정의 체인과 LLM 파이프라인 코드를 Trace와 함께 사용.


## LangChain과 함께 W&B Trace 사용

:::info
**버전** `wandb >= 0.15.4` 및 `langchain >= 0.0.218`을 사용하십시오.
:::

LangChain의 1줄 환경 변수로 W&B Trace는 LangChain Model, Chain, Agent에 대한 호출을 지속적으로 로그합니다.

또한 LangChain 문서에서 W&B Trace에 대한 문서를 확인할 수 있습니다. [LangChain documentation](https://python.langchain.com/docs/integrations/providers/wandb_tracing).

이 퀵스타트에서는 LangChain Math Agent를 사용할 것입니다:

### 1. LANGCHAIN_WANDB_TRACING 환경 변수 설정

먼저, LANGCHAIN_WANDB_TRACING 환경 변수를 true로 설정합니다. 이렇게 하면 LangChain과 함께 자동으로 Weights & Biases에 로그가 기록됩니다.

```python
import os

# langchain에 대한 wandb 로그 활성화
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
```

완료되었습니다! 이제 LangChain LLM, Chain, Tool 또는 Agent에 대한 호출이 Weights & Biases에 기록됩니다.

### 2. Weights & Biases 설정 구성
필요한 경우 추가적인 Weights & Biases [환경 변수](/guides/track/environment-variables)를 설정하여 `wandb.init()`에 일반적으로 전달되는 파라미터를 설정할 수 있습니다. 자주 사용되는 파라미터로는 W&B에서 로그가 전송되는 곳을 추가적으로 제어하기 위한 `WANDB_PROJECT`나 `WANDB_ENTITY`가 있습니다. [`wandb.init`](../../../ref/python/init.md)에 대한 자세한 정보는 API 참조 가이드를 확인하세요.

```python
# 필요에 따라 wandb 설정 또는 설정값을 설정
os.environ["WANDB_PROJECT"] = "langchain-tracing"
```


### 3. LangChain Agent 생성
LangChain을 사용하여 표준 수학 Agent를 생성하십시오:

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```


### 4. Agent 실행 및 Weights & Biases 로그 시작
일반적으로 Agent를 호출하여 LangChain을 사용하십시오. Weights & Biases run이 시작되고 Weights & Biases **[API key](https:wwww.wandb.ai/authorize)**를 입력하라는 메시지가 표시됩니다. API 키를 입력하면, Agent 호출의 입력 및 출력이 Weights & Biases App에 스트리밍되기 시작합니다.

```python
# 수학 문제 샘플
questions = [
    "5.4의 제곱근을 찾으시오.",
    "파이승 7.34를 나눈 값은 얼마인가?",
    "0.47 라디안의 사인, 27의 세제곱근으로 나눈 값은 무엇인가?",
]

for question in questions:
    try:
        # 일반적으로 Agent 호출
        answer = math_agent.run(question)
        print(answer)
    except Exception as e:
        # 모든 오류는 Weights & Biases에 또한 기록됩니다
        print(e)
        pass
```

각 Agent 실행이 완료되면 LangChain 오브젝트에 있는 모든 호출이 Weights & Biases에 기록됩니다.


### 5. Weights & Biases에서 trace 보기

이전 단계에서 생성된 W&B [run](../../runs/intro.md) 링크를 클릭하십시오. 이 링크는 W&B App의 프로젝트 워크스페이스로 리디렉션 됩니다.

생성한 run을 선택하여 trace 테이블, trace 타임라인 및 LLM의 모델 아키텍처를 확인할 수 있습니다.

![](/images/prompts/trace_timeline_detailed.png)


### 6. LangChain 컨텍스트 관리자
유스 케이스에 따라 W&B에 대한 로그 관리를 위한 컨텍스트 관리자를 사용하고 싶을 수 있습니다:

```python
from langchain.callbacks import wandb_tracing_enabled

# 환경 변수를 해제하고 대신 컨텍스트 관리자 사용
if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

# 컨텍스트 관리자를 사용하여 추적 활성화
with wandb_tracing_enabled():
    math_agent.run("What is 5 raised to .123243 power?")  # 이것은 추적되어야 함

math_agent.run("What is 2 raised to .123243 power?")  # 이것은 추적되지 않아야 함
```

이 LangChain 인테그레이션에 대한 문제는 `langchain` 태그와 함께 [wandb 저장소](https://github.com/wandb/wandb/issues)에 보고해 주세요.


## W&B Trace를 사용하여 모든 LLM 파이프라인 또는 플러그인과 함께 사용하기

:::info
**버전** `wandb >= 0.15.4`을 사용하십시오.
:::

W&B Trace는 하나 이상의 "스팬"을 로그함으로써 생성됩니다. 루트 스팬이 예상되며, 이는 중첩된 자식 스팬을 수용할 수 있으며, 이들은 다시 자신의 자식 스팬을 수용할 수 있습니다. 스팬은 `AGENT`, `CHAIN`, `TOOL` 또는 `LLM` 유형일 수 있습니다.

Trace를 사용하여 로그할 때 단일 W&B run은 LLM, Tool, Chain 또는 Agent에의 여러 호출을 로그할 수 있으며, 각 모델 또는 파이프라인 생성 후 새 W&B run을 시작할 필요 없이 각 호출이 Trace 테이블에 추가됩니다.

이 퀵스타트에서는 OpenAI 모델에 단일 호출을 단일 스팬으로 W&B Trace에 로그하는 방법을 배우고, 더 복잡한 일련의 중첩된 스팬을 로그하는 방법도 보여드립니다.

### 1. Trace 가져오기 및 Weights & Biases run 시작

`wandb.init`을 호출하여 W&B run을 시작하십시오. 여기에서는 W&B 프로젝트 이름과 엔티티 이름(팀으로 로그할 때), 설정 등을 전달할 수 있습니다. 전체 인수 목록은 [`wandb.init`](../../../ref/python/init.md)를 참조하십시오.

W&B run을 시작하면 Weights & Biases **[API 키](https:wwww.wandb.ai/authorize)**로 로그인을 요구받습니다.

```python
import wandb

# 로그할 wandb run 시작
wandb.init(project="trace-example")
```

W&B 팀에 로그할 경우 `wandb.init`의 `entity` 인수도 설정할 수 있습니다.

### 2. Trace에 로그하기
이제 OpenAI 시간을 쿼리하고 결과를 W&B Trace에 로그할 것입니다. 입력과 출력, 시작 및 종료 시간, OpenAI 호출이 성공했는지, 토큰 사용량 및 추가 메타데이터를 로그할 것입니다.

Trace 클래스의 인수에 대한 전체 설명은 [여기](https://github.com/wandb/wandb/blob/653015a014281f45770aaf43627f64d9c4f04a32/wandb/sdk/data_types/trace_tree.py#L166)에서 확인할 수 있습니다.

```python
import openai
import datetime
from wandb.sdk.data_types.trace_tree import Trace

openai.api_key = "<YOUR_OPENAI_API_KEY>"

# 구성 정의
model_name = "gpt-3.5-turbo"
temperature = 0.7
system_message = "You are a helpful assistant that always replies in 3 concise bullet points using markdown."

queries_ls = [
    "What is the capital of France?",
    "How do I boil an egg?" * 10000,  # 고의적으로 openai 오류를 발생시킨다
    "What to do if the aliens arrive?",
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
        )  # 밀리초 단위로 로그됨
        status = "success"
        status_message = (None,)
        response_text = response["choices"][0]["message"]["content"]
        token_usage = response["usage"].to_dict()

    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # 밀리초 단위로 로그됨
        status = "error"
        status_message = str(e)
        response_text = ""
        token_usage = {}

    # wandb에서 스판 생성
    root_span = Trace(
        name="root_span",
        kind="llm",  # 종류는 "llm", "chain", "agent" 또는 "tool"일 수 있음
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

    # wandb에 스판 로그
    root_span.log(name="openai_trace")
```

### 3. Weights & Biases에서 trace 보기

2단계에서 생성된 W&B [run](../../runs/intro.md) 링크를 클릭하십시오. 여기에서 LLM의 trace 테이블과 trace 타임라인을 볼 수 있어야 합니다.

### 4. 중첩된 스팬을 사용하는 LLM 파이프라인 로그
이 예에서는 Agent가 호출되는 것을 시뮬레이션한 후 LLM Chain을 호출하고, OpenAI LLM을 호출한 후 Agent가 Calculator 도구를 "호출"합니다.

"Agent" 실행의 각 단계에 대한 입력, 출력 및 메타데이터는 자체 스팬에 로그됩니다. 스팬은 자식을 가질 수 있으며

```python
import time

# 에이전트가 답해야 하는 쿼리
query = "How many days until the next US election?"

# part 1 - 에이전트가 시작됩니다...
start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

root_span = Trace(
    name="MyAgent",
    kind="agent",
    start_time_ms=start_time_ms,
    metadata={"user": "optimus_12"},
)


# part 2 - 에이전트가 LLMChain을 호출합니다...
chain_span = Trace(name="LLMChain", kind="chain", start_time_ms=start_time_ms)

# 루트의 자식으로 Chain 스팬 추가
root_span.add_child(chain_span)


# part 3 - LLMChain이 OpenAI LLM을 호출합니다...
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

# Chain 스팬의 자식으로 LLM 스팬 추가...
chain_span.add_child(llm_span)

# Chain 스팬의 끝 시간을 업데이트
chain_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"response": response_text}
)

# Chain 스팬의 끝 시간을 업데이트
chain_span._span.end_time_ms = llm_end_time_ms


# part 4 - 그 후 에이전트가 도구를 호출합니다...
time.sleep(3)
days_to_election = 117
tool_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)

# 도구 스팬 생성
tool_span = Trace(
    name="Calculator",
    kind="tool",
    status_code="success",
    start_time_ms=llm_end_time_ms,
    end_time_ms=tool_end_time_ms,
    inputs={"input": response_text},
    outputs={"result": days_to_election},
)

# 루트의 자식으로 TOOL 스팬 추가
root_span.add_child(tool_span)


# part 5 - 도구의 최종 결과가 추가됩니다
root_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"result": days_to_election}
)
root_span._span.end_time_ms = tool_end_time_ms


# part 6 - 루트 스팬을 로깅하여 모든 스팬을 W&B에 로깅
root_span.log(name="openai_trace")
```

스팬이 로깅되면 W&B App에서 Trace 테이블이 업데이트되는 것을 볼 수 있습니다.


## LlamaIndex와 함께 W&B Trace 사용

:::info
**버전** `wandb >= 0.15.4` 및 `llama-index >= 0.6.35`을 사용하십시오.
:::

가장 낮은 수준에서, LlamaIndex는 이벤트 시작/종료 개념([`CBEventTypes`](https://gpt-index.readthedocs.io/en/latest/reference/callbacks.html#llama_index.callbacks.CBEventType))을 사용하여 로그를 추적합니다. 각 이벤트에는 LLM이 생성한 쿼리와 응답에 대한 정보 또는 N 청크 생성에 사용된 문서 수 등에 대한 정보를 제공하는 페이로드가 있습니다.

상위 수준에서는, 최근에 인덱스를 쿼리할 때 내부적으로 검색, LLM 호출 등이 발생하는 경우에 연결된 이벤트의 추적 맵을 구축하는 콜백 추적 개념을 도입했습니다.

`WandbCallbackHandler`는 이 추적 맵을 시각화하고 추적하는 직관적인 방법을 제공합니다. 이벤트의 페이로드를 캡처하고 wandb에 로그로 기록합니다. 또한 총 토큰 수, 프롬프트, 컨텍스트 등의 필요한 메타데이터를 추적합니다.

게다가, 이 콜백은 W&B Artifacts에 인덱스를 업로드하고 다운로드하여 인덱스를 버전 관리할 수도 있습니다.

### 1. WandbCallbackHandler 가져오기

먼저 WandbCallbackHandler를 가져와 설정합니다. 또한 추가적인 파라미터 [`wandb.init`](../../../ref/python/init.md) 파라미터(예: W&B 프로젝트 또는 엔티티)를 전달할 수 있습니다.

W&B run이 시작되고 Weights & Biases **[API 키](https:wwww.wandb.ai/authorize)**가 필요하다는 메시지가 표시됩니다. W&B run 링크가 생성되며, 로그된 LlamaIndex 쿼리 및 데이터는 로그가 시작되면 여기에 표시됩니다.

```python
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager, WandbCallbackHandler

# WandbCallbackHandler 초기화 및 wandb.init 인수 전달
wandb_args = {"project": "llamaindex"}
wandb_callback = WandbCallbackHandler(run_args=wandb_args)

# wandb_callback을 서비스 컨텍스트에 전달
callback_manager = CallbackManager([wandb_callback])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
```

### 2. 인덱스 생성

텍스트 파일을 사용하여 간단한 인덱스를 구축할 것입니다.

```python
docs = SimpleDirectoryReader("path_to_dir").load_data()
index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
```

### 3. 인덱스 쿼리 및 Weights & Biases 로그 시작

로드된 인덱스로 문서에 대한 쿼리를 시작합니다. 인덱스에 대한 모든 호출은 자동으로 Weights & Biases에 로그됩니다.

```python
questions = [
    "What did the author do growing up?",
    "Did the author travel anywhere?",
    "What does the author love to do?",
]

query_engine = index.as_query_engine()

for q in questions:
    response = query_engine.query(q)
```

### 4. Weights & Biases에서 trace 보기

1단계에서 WandbCallbackHandler를 초기화할 때 생성된 Weights and Biases run 링크를 클릭하십시오. 그러면 W&B App의 프로젝트 워크스페이스로 이동하여 trace 테이블과 trace 타임라인을 찾을 수 있습니다.

![](/images/prompts/llama_index_trace.png)

### 5. 추적 완료

LLM 쿼리 추적이 완료되면 다음과 같이 wandb 프로세스를 닫는 것이 좋습니다:

```python
wandb_callback.finish()
```

이제 Weights & Biases를 사용하여 인덱스에 대한 쿼리를 로그할 수 있습니다. 문제가 발생하면 `llamaindex` 태그와 함께 [wandb 저장소](https://github.com/wandb/wandb/issues)에 문제를 신고하십시오.

### 6. [선택 사항] Weights & Biases Artifacts에 인덱스 데이터 저장하기
Weights & Biases [Artifacts](../../artifacts/)는 버전 관리된 데이터 및 모델 저장 제품입니다. 

당신의 인덱스를 Artifacts에 로그하고 필요할 때 사용함으로써, 특정 인덱스 버전을 기록된 Trace 출력과 연관시킬 수 있으며, 인덱스를 특정 호출에서 어떤 데이터가 인덱스에 있었는지에 대한 완전한 가시성을 보장합니다.

```python
# 'index_name'에 전달된 문자열은 아티팩트 이름이 됩니다
wandb_callback.persist_index(index, index_name="my_vector_store")
```

그런 다음 W&B 실행 페이지의 아티팩트 탭으로 이동하여 업로드된 인덱스를 확인할 수 있습니다.

**W&B Artifacts에 저장된 인덱스 사용**

Artifacts에서 인덱스를 로드하면 [`StorageContext`](https://gpt-index.readthedocs.io/en/latest/reference/storage.html)를 반환합니다. 이 저장 컨텍스트를 사용하여 LlamaIndex [로드 함수](https://gpt-index.readthedocs.io/en/latest/reference/storage/indices_save_load.html)의 함수를 사용하여 인덱스를 메모리에 로드합니다.

```python
from llama_index import load_index_from_storage

storage_context = wandb_callback.load_storage_context(
    artifact_url="<entity/project/index_name:version>"
)
index = load_index_from_storage(storage_context, service_context=service_context)
```

**참고:** [`ComposableGraph`](https://gpt-index.readthedocs.io/en/latest/reference/query/query_engines/graph_query_engine.html)의 경우, 인덱스의 루트 ID는 W&B App의 아티팩트 메타데이터 탭에서 찾을 수 있습니다.

## 다음 단계

- 기존 W&B 기능인 테이블과 Runs를 사용하여 LLM 애플리케이션 성능을 추적할 수 있습니다. 자세한 사항은 이 튜토리얼을 참조하세요:
[Tutorial: Evaluate LLM application performance](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)