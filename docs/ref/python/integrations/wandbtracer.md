
# WandbTracer

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHub에서 소스 보기](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L99-L281)

Weights and Biases에 로그하는 콜백 핸들러.

```python
WandbTracer() -> Any
```

이 핸들러는 모델 아키텍처와 실행 트레이스를 Weights and Biases에 로그합니다. 이를 통해 모든 LangChain 활동이 W&B에 로그됩니다.

| 속성 | |
| :--- | :--- |
| `always_verbose` | verbose가 False이더라도 verbose 콜백을 호출할지 여부. |
| `ignore_agent` | 에이전트 콜백을 무시할지 여부. |
| `ignore_chain` | 체인 콜백을 무시할지 여부. |
| `ignore_llm` | LLM 콜백을 무시할지 여부. |

## 메서드

### `finish`

[소스 보기](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L152-L162)

```python
@staticmethod
finish() -> None
```
모든 비동기 프로세스가 완료되고 데이터가 업로드될 때까지 기다립니다.

### `finish_run`

[소스 보기](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L202-L211)

```python
finish_run() -> None
```

W&B 데이터가 업로드될 때까지 기다립니다.

### `init`

[소스 보기](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L111-L150)

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

* **`run_args`**: (dict, optional) `wandb.init()`에 전달할 인수입니다. 제공되지 않으면, `wandb.init()`은 인수 없이 호출됩니다. 자세한 내용은 `wandb.init`를 참조하세요.
* **`include_stdout`**: (bool, optional) True인 경우, `StdOutCallbackHandler`가 핸들러 목록에 추가됩니다. LangChain을 사용할 때 유용한 정보를 stdout에 출력하기 때문에 일반적인 관행입니다.
* **`additional_handlers`**: (list, optional) LangChain 핸들러 목록에 추가할 추가 핸들러 목록입니다.

모든 LangChain 활동을 모니터링하기 위해 W&B를 사용하려면, 단순히 노트북이나 스크립트 상단에서 이 함수를 호출하세요:
```
from wandb.integration.langchain import WandbTracer
WandbTracer.init()
# ...
# 노트북 / 스크립트의 끝:
WandbTracer.finish()
```.

이것은 노트북에서와 같이 동일한 인수로 반복적으로 호출해도 안전하며, run_args가 다를 경우에만 새로운 실행을 생성합니다.

### `init_run`

[소스 보기](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L164-L200)

```python
init_run(
 run_args: Optional[WandbRunArgs] = None
) -> None
```

wandb가 초기화되지 않았다면 초기화합니다.

#### 파라미터:

* **`run_args`**: (dict, optional) `wandb.init()`에 전달할 인수입니다. 제공되지 않으면, `wandb.init()`은 인수 없이 호출됩니다. 자세한 내용은 `wandb.init`를 참조하세요.

run_args가 다를 경우에만 새로운 실행을 시작하고자 합니다. 이는 노트북 설정에서 생성되는 W&B 실행 수를 줄이는 것이 더 이상적입니다. 참고: 이 메서드를 직접 호출하는 것은 일반적이지 않습니다. 대신, `WandbTracer.init()` 메서드를 사용해야 합니다. 이 메서드는 핸들러 목록에 트레이서를 수동으로 초기화하고 추가하고 싶을 때 노출됩니다.

### `load_default_session`

[소스 보기](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L264-L267)

```python
load_default_session() -> "TracerSession"
```

기본 추적 세션을 로드하고 트레이서의 세션으로 설정합니다.

### `load_session`

[소스 보기](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L259-L262)

```python
load_session(
 session_name: str
) -> "TracerSession"
```

트레이서에서 세션을 로드합니다.

### `new_session`

```python
new_session(
 name: Optional[str] = None,
 **kwargs
) -> TracerSession
```

스레드 안전하지 않습니다. 여러 스레드에서 이 메서드를 호출하지 마십시오.

### `on_agent_action`

```python
on_agent_action(
 action: AgentAction,
 **kwargs
) -> Any
```

아무 것도 하지 않습니다.

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

체인 실행에 대한 추적을 종료합니다.

### `on_chain_error`

```python
on_chain_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

체인 실행에 대한 오류를 처리합니다.

### `on_chain_start`

```python
on_chain_start(
 serialized: Dict[str, Any],
 inputs: Dict[str, Any],
 **kwargs
) -> None
```

체인 실행에 대한 추적을 시작합니다.

### `on_llm_end`

```python
on_llm_end(
 response: LLMResult,
 **kwargs
) -> None
```

LLM 실행에 대한 추적을 종료합니다.

### `on_llm_error`

```python
on_llm_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

LLM 실행에 대한 오류를 처리합니다.

### `on_llm_new_token`

```python
on_llm_new_token(
 token: str,
 **kwargs
) -> None
```

LLM 실행에 대한 새 토큰을 처리합니다.

### `on_llm_start`

```python
on_llm_start(
 serialized: Dict[str, Any],
 prompts: List[str],
 **kwargs
) -> None
```

LLM 실행에 대한 추적을 시작합니다.

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

도구 실행에 대한 추적을 종료합니다.

### `on_tool_error`

```python
on_tool_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

도구 실행에 대한 오류를 처리합니다.

### `on_tool_start`

```python
on_tool_start(
 serialized: Dict[str, Any],
 input_str: str,
 **kwargs
) -> None
```

도구 실행에 대한 추적을 시작합니다.