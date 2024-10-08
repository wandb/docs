---
title: WandbTracer
---

[View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L99-L281)

Weights & Biases에 로그를 기록하는 콜백 핸들러입니다.

```python
WandbTracer() -> Any
```

이 핸들러는 Weights & Biases에 모델 아키텍처와 run 트레이스를 기록합니다. 이를 통해 모든 LangChain 활동이 W&B에 로그로 남게 됩니다.

| 속성 | |
| :--- | :--- |
| `always_verbose` | verbose가 False여도 콜백을 호출할지 여부. |
| `ignore_agent` | 에이전트 콜백을 무시할지 여부. |
| `ignore_chain` | 체인 콜백을 무시할지 여부. |
| `ignore_llm` | LLM 콜백을 무시할지 여부. |

## 메소드

### `finish`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L152-L162)

```python
@staticmethod
finish() -> None
```
모든 비동기 프로세스가 완료되고 데이터 업로드가 완료될 때까지 기다립니다.

### `finish_run`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L202-L211)

```python
finish_run() -> None
```

W&B 데이터가 업로드될 때까지 기다립니다.

### `init`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L111-L150)

```python
@classmethod
init(
 run_args: Optional[WandbRunArgs] = None,
 include_stdout: bool = (True),
 additional_handlers: Optional[List['BaseCallbackHandler']] = None
) -> None
```

WandbTracer를 설정하고 기본 핸들러로 만듭니다.

#### 파라미터:

* **`run_args`**: (dict, optional) `wandb.init()`에 전달할 인수. 제공되지 않으면, `wandb.init()`은 인수 없이 호출됩니다. 자세한 내용은 `wandb.init`을 참조하세요.
* **`include_stdout`**: (bool, optional) True이면, `StdOutCallbackHandler`가 핸들러 목록에 추가됩니다. 이는 LangChain 사용 시 stdout에 유용한 정보를 출력하는 일반적인 관행입니다.
* **`additional_handlers`**: (list, optional) LangChain 핸들러 목록에 추가할 추가 핸들러 목록.

LangChain의 모든 활동을 모니터링하기 위해, 노트북이나 스크립트의 상단에서 이 함수를 호출합니다:
```
from wandb.integration.langchain import WandbTracer
WandbTracer.init()
# ...
# 노트북 / 스크립트의 끝:
WandbTracer.finish()
```.

같은 인수로 여러 번 호출해도 안전합니다 (예: 노트북에서). 이는 run_args가 다를 경우에만 새로운 run을 생성하기 때문입니다.

### `init_run`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L164-L200)

```python
init_run(
 run_args: Optional[WandbRunArgs] = None
) -> None
```

wandb가 초기화되지 않았으면 초기화합니다.

#### 파라미터:

* **`run_args`**: (dict, optional) `wandb.init()`에 전달할 인수. 제공되지 않으면, `wandb.init()`은 인수 없이 호출됩니다. 자세한 내용은 `wandb.init`을 참조하세요.

우리는 run 인수가 다를 때만 새로운 run을 시작하고 싶습니다. 이는 생성되는 W&B runs의 수를 줄여줍니다. 이는 노트북 환경에서 더 이상적입니다. 참고로, 이 메소드를 직접 호출하는 일은 드뭅니다. 대신 `WandbTracer.init()` 메소드를 사용하세요. 이 메소드는 트레이서를 수동으로 초기화하여 핸들러 목록에 추가하고자 할 때 노출됩니다.

### `load_default_session`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L264-L267)

```python
load_default_session() -> "TracerSession"
```

기본 트레이싱 세션을 로드하고 이를 Tracer의 세션으로 설정합니다.

### `load_session`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L259-L262)

```python
load_session(
 session_name: str
) -> "TracerSession"
```

트레이서로부터 세션을 로드합니다.

### `new_session`

```python
new_session(
 name: Optional[str] = None,
 **kwargs
) -> TracerSession
```

스레드 안전하지 않으므로 이 메소드를 여러 스레드에서 호출하지 마세요.

### `on_agent_action`

```python
on_agent_action(
 action: AgentAction,
 **kwargs
) -> Any
```

아무것도 하지 않습니다.

### `on_agent_finish`

```python
on_agent_finish(
 finish: AgentFinish,
 **kwargs
) -> None
```

에이전트 종료 메시지를 처리합니다.

### `on_chain_end`

```python
on_chain_end(
 outputs: Dict[str, Any],
 **kwargs
) -> None
```

체인 run의 트레이스를 종료합니다.

### `on_chain_error`

```python
on_chain_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

체인 run의 오류를 처리합니다.

### `on_chain_start`

```python
on_chain_start(
 serialized: Dict[str, Any],
 inputs: Dict[str, Any],
 **kwargs
) -> None
```

체인 run의 트레이스를 시작합니다.

### `on_llm_end`

```python
on_llm_end(
 response: LLMResult,
 **kwargs
) -> None
```

LLM run의 트레이스를 종료합니다.

### `on_llm_error`

```python
on_llm_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

LLM run의 오류를 처리합니다.

### `on_llm_new_token`

```python
on_llm_new_token(
 token: str,
 **kwargs
) -> None
```

LLM run의 새로운 토큰을 처리합니다.

### `on_llm_start`

```python
on_llm_start(
 serialized: Dict[str, Any],
 prompts: List[str],
 **kwargs
) -> None
```

LLM run의 트레이스를 시작합니다.

### `on_text`

```python
on_text(
 text: str,
 **kwargs
) -> None
```

텍스트 메시지를 처리합니다.

### `on_tool_end`

```python
on_tool_end(
 output: str,
 **kwargs
) -> None
```

툴 run의 트레이스를 종료합니다.

### `on_tool_error`

```python
on_tool_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

툴 run의 오류를 처리합니다.

### `on_tool_start`

```python
on_tool_start(
 serialized: Dict[str, Any],
 input_str: str,
 **kwargs
) -> None
```

툴 run의 트레이스를 시작합니다.