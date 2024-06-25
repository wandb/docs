---
description: Prompts クイックスタートでは、LLM チェーンやパイプラインの実行フローを視覚化およびデバッグする方法を示しています
displayed_sidebar: default
---


# Prompts Quickstart

[**Try in a Colab Notebook here →**](http://wandb.me/prompts-quickstart)

<head>
  <title>Prompts Quickstart</title>
</head>

このクイックスタートガイドでは、[Trace](intro.md) を使用して LangChain、LlamaIndex、または独自の LLM チェーンやパイプラインへの呼び出しを可視化およびデバッグする方法を紹介します：

1. **[Langchain:](#use-wb-trace-with-langchain)** LangChain の環境変数またはコンテキストマネージャーを使用して自動ロギングを行います。

2. **[LlamaIndex:](#use-wb-trace-with-llamaindex)** LlamaIndex の W&B コールバックを使用して自動ロギングを行います。

3. **[Custom usage](#use-wb-trace-with-any-llm-pipeline-or-plug-in)**: 独自のカスタムチェーンや LLM パイプラインコードで Trace を使用します。

## Use W&B Trace with LangChain

:::info
**バージョン** `wandb >= 0.15.4` および `langchain >= 0.0.218` を使用してください
:::

LangChain の 1 行の環境変数を使用して、W&B Trace は LangChain Model、Chain、または Agent への呼び出しを継続的にログに記録します。

W&B Trace の詳細なドキュメントについては、[LangChain documentation](https://python.langchain.com/docs/integrations/providers/wandb_tracing) もご覧ください。

このクイックスタートでは、LangChain Math Agent を使用します：

### 1. LANGCHAIN_WANDB_TRACING 環境変数を設定する

まず、LANGCHAIN_WANDB_TRACING 環境変数を true に設定します。これにより、LangChain との Weights & Biases ロギングが自動的にオンになります：

```python
import os

# LangChain 用の wandb ロギングをオンにします
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
```

これで完了です！ LangChain LLM、Chain、Tool、または Agent へのすべての呼び出しが Weights & Biases にログされます。

### 2. Weights & Biasesの設定を行う
追加の Weights & Biases [Environment Variables](/guides/track/environment-variables) を設定して、通常 `wandb.init()` に渡されるパラメータを設定できます。よく使用されるパラメータには、ロギング先をより詳細にコントロールするために `WANDB_PROJECT` や `WANDB_ENTITY` などがあります。[`wandb.init`](../../ref/python/init.md) についての詳細は API リファレンスガイドをご覧ください。

```python
# 必要に応じて wandb の設定やコンフィグを行います
os.environ["WANDB_PROJECT"] = "langchain-tracing"
```

### 3. LangChain エージェントを作成する
LangChain を使用して標準的な Math エージェントを作成します：

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 4. エージェントを実行して Weights & Biases ロギングを開始する
エージェントを呼び出して LangChain を通常通り使用します。Weights & Biases の run が開始され、Weights & Biases の **[APIキー](https:wwww.wandb.ai/authorize)** が求められます。APIキーを入力すると、エージェント呼び出しの入力と出力が Weights & Biases アプリにストリーミングされ始めます。

```python
# いくつかの数学の質問サンプル
questions = [
    "5.4 の平方根は何ですか？",
    "3 を 7.34 の累乗に割ると何になりますか？",
    "0.47 ラジアンの正弦は何ですか？（それを27の立方根で割る）",
]

for question in questions:
    try:
        # 通常通りにエージェントを呼び出します
        answer = math_agent.run(question)
        print(answer)
    except Exception as e:
        # すべてのエラーも Weights & Biases にログされます
        print(e)
        pass
```

各エージェントの実行が完了すると、LangChain オブジェクト内のすべての呼び出しが Weights & Biases にログされます。

### 5. Weights & Biases でトレースを表示する

前のステップで生成された W&B [run](../runs/intro.md) リンクをクリックします。これにより、W&B アプリ内のプロジェクトワークスペースにリダイレクトされます。

作成した run を選択して、トレーステーブル、トレースタイムライン、LLM のモデルアーキテクチャを表示します。

![](/images/prompts/trace_timeline_detailed.png)

### 6. LangChain コンテキストマネージャー
ユースケースに応じて、W&B へのロギングを管理するためにコンテキストマネージャーを使用する方が良いかもしれません：

```python
from langchain.callbacks import wandb_tracing_enabled

# 環境変数を解除し、代わりにコンテキストマネージャーを使用します
if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

# コンテキストマネージャーを使用してトレースを有効にします
with wandb_tracing_enabled():
    math_agent.run("5を0.123243乗にしたものは？")  # これはトレースされます

math_agent.run("2を0.123243乗にしたものは？")  # これはトレースされません
```

この LangChain インテグレーションに関する問題は、`langchain` タグを付けて [wandb repo](https://github.com/wandb/wandb/issues)に報告してください。

## Use W&B Trace with Any LLM Pipeline or Plug-In

:::info
**バージョン** `wandb >= 0.15.4` を使用してください
:::

W&B Trace は 1 つ以上の "span" をログ記録することにより作成されます。root span が期待され、その下にネストされた子 span を受け入れ、その子 span もさらに子 span を受け入れることができます。span のタイプは `AGENT`、`CHAIN`、`TOOL`、または `LLM` です。

Trace を使用してログ記録する場合、単一の W&B run に複数の LLM、Tool、Chain、または Agent への呼び出しをログできます。モデルやパイプラインからの各生成後に新しい W&B run を開始する必要はなく、各呼び出しは Trace テーブルに追加されます。

このクイックスタートでは、まず OpenAI モデルへの単一の呼び出しを W&B Trace に単一の span としてログする方法を示し、その後、より複雑なネストされた span のシリーズをログする方法を示します。

### 1. Trace をインポートし、Weights & Biases run を開始する

`wandb.init` を呼び出して W&B run を開始します。ここでは、W&B プロジェクト名やエンティティ名（W&B Teamにログを送信する場合）だけでなく、コンフィグなども渡すことができます。引数の完全なリストについては、[`wandb.init`](../../ref/python/init.md)をご覧ください。

W&B run を開始すると、Weights & Biases の **[APIキー](https:wwww.wandb.ai/authorize)** が求められます。

```python
import wandb

# wandb run を開始してログに記録する
wandb.init(project="trace-example")
```

また、W&B Teamにログを送信する場合は、`wandb.init` の `entity` 引数を設定することもできます。

### 2. Trace にログする
次に、OpenAI にクエリーを送信し、その結果を W&B Trace にログします。入力と出力、開始および終了時刻、OpenAI 呼び出しが成功したかどうか、トークンの使用量、追加のメタデータをログします。

Trace クラスの引数の完全な説明は[こちら](https://github.com/wandb/wandb/blob/653015a014281f45770aaf43627f64d9c4f04a32/wandb/sdk/data_types/trace_tree.py#L166)をご覧ください。

```python
import openai
import datetime
from wandb.sdk.data_types.trace_tree import Trace

openai.api_key = "<YOUR_OPENAI_API_KEY>"

# コンフィグを定義する
model_name = "gpt-3.5-turbo"
temperature = 0.7
system_message = "You are a helpful assistant that always replies in 3 concise bullet points using markdown."

queries_ls = [
    "フランスの首都はどこですか？",
    "卵の茹で方を教えてください？" * 10000,  # 故意に openai エラーを発生させる
    "エイリアンが到着したらどうしますか？",
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
        )  # ミリ秒でログを記録する
        status = "success"
        status_message = (None,)
        response_text = response["choices"][0]["message"]["content"]
        token_usage = response["usage"].to_dict()

    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # ミリ秒でログを記録する
        status = "error"
        status_message = str(e)
        response_text = ""
        token_usage = {}

    # wandb 内で span を作成する
    root_span = Trace(
        name="root_span",
        kind="llm",  # kind は "llm", "chain", "agent", "tool" のいずれか
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

    # span を wandb にログする
    root_span.log(name="openai_trace")
```

### 3. Weights & Biasesでトレースを表示する

ステップ 2 で生成された W&B [run](../runs/intro.md) リンクをクリックします。ここで、トレーステーブルおよびトレースタイムラインを見ることができます。

### 4. ネストされた spans を使用して LLM パイプラインをログする
この例では、エージェントが呼び出され、それから LLM チェーンが呼び出され、OpenAI LLM が呼び出され、最後にエージェントが計算ツールを呼び出すプロセスをシミュレートします。

エージェントの各ステップの入力、出力、メタデータがそれぞれの span に記録されます。spans は子 spans を持つことができます。

```python
import time

# エージェントが回答する必要のあるクエリ
query = "次の米国選挙まであと何日ですか？"

# part 1 - エージェントが開始されます...
start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

root_span = Trace(
    name="MyAgent",
    kind="agent",
    start_time_ms=start_time_ms,
    metadata={"user": "optimus_12"},
)

# part 2 - エージェントが LLMChain に呼び出します...
chain_span = Trace(name="LLMChain", kind="chain", start_time_ms=start_time_ms)

# root に Chain を子として追加
root_span.add_child(chain_span)

# part 3 - LLMChain が OpenAI LLM を呼び出します...
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

# LLM span を Chain span の子として追加...
chain_span.add_child(llm_span)

# Chain span の終了時刻を更新
chain_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"response": response_text}
)

# Chain span の終了時刻を更新
chain_span._span.end_time_ms = llm_end_time_ms

# part 4 - エージェントがツールを呼び出します...
time.sleep(3)
days_to_election = 117
tool_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)

# Tool span を作成する
tool_span = Trace(
    name="Calculator",
    kind="tool",
    status_code="success",
    start_time_ms=llm_end_time_ms,
    end_time_ms=tool_end_time_ms,
    inputs={"input": response_text},
    outputs={"result": days_to_election},
)

# TOOL span を root の子として追加
root_span.add_child(tool_span)

# part 5 - 最終的な結果をツールから追加する
root_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"result": days_to_election}
)
root_span._span.end_time_ms = tool_end_time_ms

# part 6 - root span をログしてすべての spans を W&B にログする
root_span.log(name="openai_trace")
```

span をログしたら、W&B アプリで Trace テーブルが更新されるのが確認できます。

## Use W&B Trace with LlamaIndex

:::info
**バージョン** `wandb >= 0.15.4` および `llama-index >= 0.6.35` を使用してください
:::

LlamaIndex の最低レベルでは、start/end イベント([`CBEventTypes`](https://gpt-index.readthedocs.io/en/latest/reference/callbacks.html#llama_index.callbacks.CBEventType))のコンセプトを使用してログを追跡します。各イベントには、LLM によって生成されたクエリや応答、Nチャンクを作成するために使用されたドキュメントの数などの情報を提供するペイロードがあります。

より高いレベルでは、最近コールバックトレースの概念が導入され、接続されたイベントのトレースマップを構築します。例えば、インデックスに対してクエリを実行すると、その背後で取得、LLM の呼び出しなどが行われます。

`WandbCallbackHandler` は、このトレースマップを直感的に可視化し、追跡する方法を提供します。イベントのペイロードをキャプチャして wandb にログし、合計トークン数、プロンプト、コンテキストなどの必要なメタデータも追跡します。

さらに、このコールバックを使用してインデックスを W&B Artifacts にアップロードおよびダウンロードすることで、インデックスのバージョン管理を行うこともできます。

### 1. WandbCallbackHandler をインポートする

まず `WandbCallbackHandler` をインポートして設定します。また、W&B プロジェクトやエンティティなどの追加パラメータを [`wandb.init`](../../ref/python/init.md) に渡すこともできます。

W&B run が開始され、Weights & Biases の **[APIキー](https:wwww.wandb.ai/authorize)** が求められます。W&B run リンクが生成され、ここでログされた LlamaIndex クエリとデータを確認できます。

```python
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager, WandbCallbackHandler

# WandbCallbackHandler を初期化し、wandb.init 引数を渡す
wandb_args = {"project": "llamaindex"}
wandb_callback = WandbCallbackHandler(run_args=wandb_args)

# wandb_callback をサービスコンテキストに渡す
callback_manager = CallbackManager([wandb_callback])
service_context = ServiceContext.from