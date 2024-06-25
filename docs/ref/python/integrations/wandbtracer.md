
# WandbTracer

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L99-L281)

Weights and Biases にログするコールバックハンドラー。

```python
WandbTracer() -> Any
```

このハンドラーはモデルアーキテクチャーと run のトレースを Weights and Biases にログします。これにより、すべての LangChain の活動が W&B にログされるようになります。

| 属性 | 説明 |
| :--- | :--- |
| `always_verbose` | verbose が False の場合でも verbose コールバックを呼び出すかどうか。 |
| `ignore_agent` | エージェントコールバックを無視するかどうか。 |
| `ignore_chain` | チェーンコールバックを無視するかどうか。 |
| `ignore_llm` | LLM コールバックを無視するかどうか。 |

## メソッド

### `finish`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L152-L162)

```python
@staticmethod
finish() -> None
```
すべての非同期プロセスが終了し、データがアップロードされるのを待ちます。

### `finish_run`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L202-L211)

```python
finish_run() -> None
```

W&B データがアップロードされるのを待ちます。

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

WandbTracer をセットアップし、デフォルトハンドラーにします。

#### パラメータ:

* **`run_args`**: (dict, optional) `wandb.init()` に渡す引数。指定しない場合、`wandb.init()` は引数なしで呼び出されます。詳細は `wandb.init` を参照してください。
* **`include_stdout`**: (bool, optional) True の場合、`StdOutCallbackHandler` がハンドラーのリストに追加されます。これは、LangChain を使用する際に標準出力に有用な情報を表示するため、一般的なプラクティスです。
* **`additional_handlers`**: (list, optional) LangChain ハンドラーのリストに追加する追加ハンドラーのリスト。

LangChain のすべての活動を監視するためには、この関数をノートブックやスクリプトの冒頭で呼び出してください:
```
from wandb.integration.langchain import WandbTracer
WandbTracer.init()
# ...
# end of notebook / script:
WandbTracer.finish()
```.

同じ引数で繰り返し呼び出しても安全です（ノートブック内などで）。引数が異なる場合にのみ新しい run が作成されます。

### `init_run`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L164-L200)

```python
init_run(
 run_args: Optional[WandbRunArgs] = None
) -> None
```

wandb が初期化されていない場合に初期化します。

#### パラメータ:

* **`run_args`**: (dict, optional) `wandb.init()` に渡す引数。指定しない場合、`wandb.init()` は引数なしで呼び出されます。詳細は `wandb.init` を参照してください。

run の引数が異なる場合にのみ新しい run を開始することを望みます。これにより、ノートブック環境で作成される W&B run の数が減り、より理想的な状況になります。注: このメソッドを直接呼び出すことは一般的ではありません。代わりに、`WandbTracer.init()` メソッドを使用するべきです。このメソッドは、トレーサーを手動で初期化し、ハンドラーのリストに追加したい場合に公開されています。

### `load_default_session`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L264-L267)

```python
load_default_session() -> "TracerSession"
```

デフォルトのトレーシングセッションをロードし、トレーサーのセッションとして設定します。

### `load_session`

[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L259-L262)

```python
load_session(
 session_name: str
) -> "TracerSession"
```

トレーサーからセッションをロードします。

### `new_session`

```python
new_session(
 name: Optional[str] = None,
 **kwargs
) -> TracerSession
```

スレッドセーフではありません。複数のスレッドからこのメソッドを呼び出さないでください。

### `on_agent_action`

```python
on_agent_action(
 action: AgentAction,
 **kwargs
) -> Any
```

何もしません。

### `on_agent_finish`

```python
on_agent_finish(
 finish: AgentFinish,
 **kwargs
) -> None
```

エージェント終了メッセージを処理します。

### `on_chain_end`

```python
on_chain_end(
 outputs: Dict[str, Any],
 **kwargs
) -> None
```

チェーン run のトレースを終了します。

### `on_chain_error`

```python
on_chain_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

チェーン run のエラーを処理します。

### `on_chain_start`

```python
on_chain_start(
 serialized: Dict[str, Any],
 inputs: Dict[str, Any],
 **kwargs
) -> None
```

チェーン run のトレースを開始します。

### `on_llm_end`

```python
on_llm_end(
 response: LLMResult,
 **kwargs
) -> None
```

LLM run のトレースを終了します。

### `on_llm_error`

```python
on_llm_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

LLM run のエラーを処理します。

### `on_llm_new_token`

```python
on_llm_new_token(
 token: str,
 **kwargs
) -> None
```

LLM run の新しいトークンを処理します。

### `on_llm_start`

```python
on_llm_start(
 serialized: Dict[str, Any],
 prompts: List[str],
 **kwargs
) -> None
```

LLM run のトレースを開始します。

### `on_text`

```python
on_text(
 text: str,
 **kwargs
) -> None
```

テキストメッセージを処理します。

### `on_tool_end`

```python
on_tool_end(
 output: str,
 **kwargs
) -> None
```

ツール run のトレースを終了します。

### `on_tool_error`

```python
on_tool_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

ツール run のエラーを処理します。

### `on_tool_start`

```python
on_tool_start(
 serialized: Dict[str, Any],
 input_str: str,
 **kwargs
) -> None
```

ツール run のトレースを開始します。