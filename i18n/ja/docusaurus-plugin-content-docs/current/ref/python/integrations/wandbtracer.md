# WandbTracer

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L99-L281)

Weights and Biasesにログを送信するコールバックハンドラ。

```python
WandbTracer() -> Any
```

このハンドラは、モデルのアーキテクチャーとrunのトレースをWeights and Biasesにログします。このクラスを直接インスタンス化する必要があることはまれです。代わりに、`WandbTracer.init()` メソッドを使用してハンドラを設定し、デフォルトのハンドラにする必要があります。これにより、すべてのLangChainのアクティビティがW&Bにログされることが保証されます。

| 属性 | |
| :--- | :--- |
| `always_verbose` | verboseがFalseであっても、verboseコールバックを呼び出します。 |
| `ignore_agent` | エージェントのコールバックを無視するかどうか。 |
| `ignore_chain` | チェーンのコールバックを無視するかどうか。 |
| `ignore_llm` | LLMのコールバックを無視するかどうか。 |
## メソッド

### `finish`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L152-L162)

```python
@staticmethod
finish() -> None
```

すべてのLangChainのアクティビティの監視を停止し、デフォルトのハンドラをリセットします。

カーネルやPythonスクリプトを終了する前に、この関数を呼び出すことが推奨されます。

### `finish_run`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L202-L211)

```python
finish_run() -> None
```
W&Bデータのアップロード待ち。

### `init`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L111-L150)

```python
@classmethod
init(
 run_args: Optional[WandbRunArgs] = None,
 include_stdout: bool = (True),
 additional_handlers: Optional[List['BaseCallbackHandler']] = None
) -> None
```

WandbTracerを設定し、デフォルトのハンドラーにします。

#### パラメータ:

* **`run_args`**: (dict, optional) `wandb.init()`に渡す引数。提供されていない場合、`wandb.init()`は引数なしで呼び出されます。詳細については、`wandb.init`を参照してください。
* **`include_stdout`**: (bool, optional) Trueの場合、`StdOutCallbackHandler`がハンドラーのリストに追加されます。これは、LangChainを使用する際によく行われる実践であり、stdoutに役立つ情報を出力します。。
* **`additional_handlers`**: (list, optional) LangChainハンドラーのリストに追加する追加ハンドラーのリスト。
W&Bを使用してLangChainのすべてのアクティビティを監視するには、ノートブックまたはスクリプトの先頭でこの関数を呼び出すだけです:
```
from wandb.integration.langchain import WandbTracer
WandbTracer.init()
# ...
# ノートブック / スクリプトの終わり:
WandbTracer.finish()
```

同じ引数を使用して何度も呼び出しても問題ありません（ノートブック内など）、run_argsが異なる場合にのみ新しいrunが作成されます。

### `init_run`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L164-L200)

```python
init_run(
 run_args: Optional[WandbRunArgs] = None
) -> None
```

wandbが初期化されていない場合は、wandbを初期化します。
#### パラメータ:


* **`run_args`**: (辞書, オプション) `wandb.init()`に渡す引数。指定しない場合、`wandb.init()`は引数なしで呼び出されます。詳細については、`wandb.init`を参照してください。

run_argsが異なる場合にのみ、新しいrunを開始したいと考えています。これにより、ノートブックの環境でより理想的な、W&B runsの数が減ります。注：このメソッドを直接呼び出すことは珍しいです。代わりに、`WandbTracer.init()`メソッドを使用するべきです。このメソッドは、トレーサーを手動で初期化し、ハンドラーのリストに追加する場合に公開されています。

### `load_default_session`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L264-L267)

```python
load_default_session() -> "TracerSession"
```

デフォルトのトレーシングセッションをロードし、トレーサーのセッションとして設定します。


### `load_session`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L259-L262)

```python
load_session(
 session_name: str
) -> "TracerSession"
```

トレーサからセッションを読み込みます。


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
何もしないでください。

### `on_agent_finish`

```python
on_agent_finish(
 finish: AgentFinish,
 **kwargs
) -> None
```

エージェント完了メッセージを処理します。

### `on_chain_end`

```python
on_chain_end(
 outputs: Dict[str, Any],
 **kwargs
) -> None
```

チェーン実行のトレースを終了します。

### `on_chain_error`

```python
on_chain_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

チェーンrunのエラーを処理する。

### `on_chain_start`

```python
on_chain_start(
 serialized: Dict[str, Any],
 inputs: Dict[str, Any],
 **kwargs
) -> None
```

チェーンrunのトレースを開始する。

### `on_llm_end`

```python
on_llm_end(
 response: LLMResult,
 **kwargs
) -> None
```

LLMの実行に関するトレースを終了させる。

### `on_llm_error`

```python
on_llm_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

LLMの実行中に発生したエラーを処理する。

### `on_llm_new_token`

```python
on_llm_new_token(
 トークン: str,
 **kwargs
) -> None
```

LLMのrunのための新しいトークンを処理します。


### `on_llm_start`



```python
on_llm_start(
 シリアライズされた: Dict[str, Any],
 プロンプト: List[str],
 **kwargs
) -> None
```

LLMのrunのためにトレースを開始します。


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

ツールのrunのトレースを終了します。


### `on_tool_error`



```python
on_tool_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```
ツールの実行に対するエラーを処理します。



### `on_tool_start`



```python

on_tool_start(

 serialized: Dict[str, Any],

 input_str: str,

 **kwargs

) -> None

```



ツールの実行に対するトレースを開始します。