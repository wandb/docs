---
description: LLMチェーンとパイプラインの実行フローを可視化し、デバッグする方法を示しています。
displayed_sidebar: default
---


# Prompts クイックスタート

[**Colab ノートブックで試す →**](http://wandb.me/prompts-quickstart)

<head>
  <title>Prompts クイックスタート</title>
</head>

このクイックスタートガイドでは、[Trace](intro.md) を使用して LangChain、LlamaIndex、または独自の LLM チェインやパイプラインの呼び出しを可視化およびデバッグする方法を説明します：

1. **[Langchain:](#use-wb-trace-with-langchain)** 自動ログ記録のための LangChain の 1 行の環境変数またはコンテキストマネージャーインテグレーションを使用します。

2. **[LlamaIndex:](#use-wb-trace-with-llamaindex)** LlamaIndex からの W&B コールバックを使用して自動ログ記録を行います。

3. **[カスタム使用法](#use-wb-trace-with-any-llm-pipeline-or-plug-in)**: 独自のカスタムチェーンや LLM パイプラインコードと一緒に Trace を使用します。

## LangChain で W&B Trace を使用する

:::info
**バージョン** `wandb >= 0.15.4` および `langchain >= 0.0.218` を使用してください。
:::

LangChain の 1 行の環境変数を使用すると、W&B Trace は LangChain モデル、チェーン、またはエージェントへの呼び出しを継続的にログ記録します。

詳細については、[LangChain ドキュメント](https://python.langchain.com/docs/integrations/providers/wandb_tracing)をご覧ください。

このクイックスタートでは、LangChain Math Agent を使用します：

### 1. LANGCHAIN_WANDB_TRACING 環境変数の設定

まず、LANGCHAIN_WANDB_TRACING 環境変数を true に設定します。これにより、LangChain での自動 Weights & Biases ログ記録がオンになります：

```python
import os

# LangChain 用の wandb ログ記録をオンにする
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
```

これで完了です！ LangChain の LLM、チェーン、ツール、またはエージェントへのあらゆる呼び出しが Weights & Biases にログ記録されます。

### 2. Weights & Biases 設定の構成
追加の Weights & Biases [環境変数](/guides/track/environment-variables) を設定して、通常 `wandb.init()` に渡すパラメータを設定することもできます。よく使用されるパラメータには、`WANDB_PROJECT` や `WANDB_ENTITY` などがあります。詳細については、[`wandb.init`](../../ref/python/init.md) の API リファレンスガイドをご覧ください。

```python
# 任意で wandb 設定や構成を設定する
os.environ["WANDB_PROJECT"] = "langchain-tracing"
```

### 3. LangChain エージェントを作成する
LangChain を使用して標準の数学エージェントを作成します：

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 4. エージェントを実行して Weights & Biases のログ記録を開始する
エージェントを呼び出して LangChain を通常通り使用します。Weights & Biases の run が開始され、Weights & Biases **[API キー](https:wwww.wandb.ai/authorize)** を求められます。API キーを入力すると、エージェント呼び出しの入力と出力が Weights & Biases アプリにストリームされ始めます。

```python
# いくつかのサンプル数学クエスチョン
questions = [
    "5.4 の平方根を求めてください。",
    "3 を 7.34 の pi 乗で割ると？",
    "0.47 ラジアンの sin を 27 の立方根で割ると？",
]

for question in questions:
    try:
        # 通常通りエージェントを呼び出す
        answer = math_agent.run(question)
        print(answer)
    except Exception as e:
        # すべてのエラーも Weights & Biases にログ記録されます
        print(e)
        pass
```

各エージェント実行が完了すると、LangChain オブジェクト内のすべての呼び出しが Weights & Biases にログ記録されます。

### 5. Weights & Biases でトレースを表示する

前のステップで生成された W&B [run](../runs/intro.md) リンクをクリックします。これにより、W&B アプリのプロジェクトワークスペースにリダイレクトされます。

作成した run を選択し、トレーステーブル、トレースタイムライン、LLM のモデルアーキテクチャを表示します。

![](/images/prompts/trace_timeline_detailed.png)

### 6. LangChain コンテキストマネージャー
ユースケースによっては、ログ記録を W&B に管理するためにコンテキストマネージャーを使用する方が適しているかもしれません：

```python
from langchain.callbacks import wandb_tracing_enabled

# 環境変数を解除し、代わりにコンテキストマネージャーを使用する
if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

# コンテキストマネージャーを使用してトレースを有効にする
with wandb_tracing_enabled():
    math_agent.run("5 の 0.123243 乗は？")  # これがトレースされます

math_agent.run("2 の 0.123243 乗は？")  # これはトレースされません
```

この LangChain インテグレーションに関する問題は、`langchain` タグを付けて [wandb リポジトリ](https://github.com/wandb/wandb/issues) に報告してください。

## 任意の LLM パイプラインまたはプラグインで W&B Trace を使用する

:::info
**バージョン** `wandb >= 0.15.4` を使用してください。
:::

W&B Trace は 1 つ以上の "スパン" をログ記録することで作成されます。ルートスパンが期待され、その中にネストされた子スパンを受け入れることができ、その子スパンもさらに独自の子スパンを受け入れます。スパンは `AGENT`、`CHAIN`、`TOOL`、または `LLM` のタイプである可能性があります。

Trace を使用してログ記録する場合、単一の W&B run に対して複数の LLM、ツール、チェーン、またはエージェントの呼び出しをログ記録でき、新しい W&B run をモデルまたはパイプラインの各生成後に開始する必要はありません。代わりに、各呼び出しが Trace テーブルに追加されます。

このクイックスタートでは、OpenAI モデルへの単一の呼び出しを W&B Trace に単一スパンとしてログ記録する方法を説明します。その後、ネストされたスパンのより複雑な一連のログ記録方法を示します。

### 1. Trace と Weights & Biases run のインポートと開始

`wandb.init` を呼び出して W&B run を開始します。ここで W&B プロジェクト名やエンティティ名（W&B チームにログ記録する場合）、設定などを渡すことができます。引数の全リストについては [`wandb.init`](../../ref/python/init.md) を参照してください。

W&B run を開始すると、Weights & Biases **[API キー](https:wwww.wandb.ai/authorize)** でログインするように求められます。

```python
import wandb

# ログ記録のために wandb run を開始
wandb.init(project="trace-example")
```

W&B チームにログ記録する場合は、`wandb.init` に `entity` 引数を設定することもできます。

### 2. Trace へのログ記録
次に、OpenAI に問い合わせて結果を W&B Trace にログ記録します。入力と出力、開始と終了の時間、OpenAI 呼び出しの成功可否、トークン使用量、追加のメタデータをログ記録します。

Trace クラスの引数の完全な説明は [こちら](https://github.com/wandb/wandb/blob/653015a014281f45770aaf43627f64d9c4f04a32/wandb/sdk/data_types/trace_tree.py#L166) で確認できます。

```python
import openai
import datetime
from wandb.sdk.data_types.trace_tree import Trace

openai.api_key = "<YOUR_OPENAI_API_KEY>"

# 設定を定義
model_name = "gpt-3.5-turbo"
temperature = 0.7
system_message = "You are a helpful assistant that always replies in 3 concise bullet points using markdown."

queries_ls = [
    "フランスの首都はどこですか？",
    "卵を茹でるにはどうすればよいですか？" * 10000,  # deliberately trigger an openai error
    "宇宙人が来たらどうしますか？",
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
        )  # ミリ秒でログ記録
        status = "success"
        status_message = (None,)
        response_text = response["choices"][0]["message"]["content"]
        token_usage = response["usage"].to_dict()

    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # ミリ秒でログ記録
        status = "error"
        status_message = str(e)
        response_text = ""
        token_usage = {}

    # wandb にスパンを作成
    root_span = Trace(
        name="root_span",
        kind="llm",  # 種類は "llm", "chain", "agent" または "tool"
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

    # span を wandb にログ記録
    root_span.log(name="openai_trace")
```

### 3. Weights & Biases でトレースを表示する

ステップ 2 で生成された W&B [run](../runs/intro.md) リンクをクリックします。ここでは、LLM のトレーステーブルやトレースタイムラインを表示できます。

### 4. ネストされたスパンを使用して LLM パイプラインをログ記録する
この例では、エージェントが呼び出され、その後 LLM チェーンが呼び出され、OpenAI LLM が呼び出され、最終的にエージェントが "Calculator" ツールを呼び出すシナリオをシミュレートします。

エージェントの各ステップの入力、出力、メタデータはそれぞれ独自のスパンにログ記録されます。スパンは子スパンを持つことができます。

```python
import time

# エージェントが答えるべきクエリ
query = "次の米国選挙まで何日ですか？"

# パート 1 - エージェントが開始される...
start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

root_span = Trace(
    name="MyAgent",
    kind="agent",
    start_time_ms=start_time_ms,
    metadata={"user": "optimus_12"},
)


# パート 2 - エージェントが LLMChain を呼び出す...
chain_span = Trace(name="LLMChain", kind="chain", start_time_ms=start_time_ms)

# チェーンスパンをルートに子として追加
root_span.add_child(chain_span)


# パート 3 - LLMChain が OpenAI LLM を呼び出す...
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

# LLM スパンをチェーンスパンの子として追加...
chain_span.add_child(llm_span)

# チェーンスパンの終了時間を更新
chain_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"response": response_text}
)

# チェーンスパンの終了時間を更新
chain_span._span.end_time_ms = llm_end_time_ms


# パート 4 - エージェントがツールを呼び出す...
time.sleep(3)
days_to_election = 117
tool_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)

# ツールスパンを作成
tool_span = Trace(
    name="Calculator",
    kind="tool",
    status_code="success",
    start_time_ms=llm_end_time_ms,
    end_time_ms=tool_end_time_ms,
    inputs={"input": response_text},
    outputs={"result": days_to_election},
)

# TOOL スパンをルートの子として追加
root_span.add_child(tool_span)


# パート 5 - ツールからの最終結果を追加
root_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"result": days_to_election}
)
root_span._span.end_time_ms = tool_end_time_ms


# パート 6 - ルートスパンをログすることで、すべてのスパンを W&B に記録する
root_span.log(name="openai_trace")
```

スパンをログ記録すると、W&B アプリで Trace テーブルが更新されたのが見えるようになります。

## LlamaIndex で W&B Trace を使用する

:::info
**バージョン** `wandb >= 0.15.4` および `llama-index >= 0.6.35` を使用してください。
:::

最も低いレベルでは、LlamaIndex は開始/終了イベント（[`CBEventTypes`](https://gpt-index.readthedocs.io/en/latest/reference/callbacks.html#llama_index.callbacks.CBEventType)）の概念を使用して、ログの追跡を行います。各イベントにはペイロードがあり、LLM が生成したクエリや応答、使用されたドキュメント数などの情報を提供します。

より高いレベルでは、最近、コールバックトレースの概念が導入され、接続されたイベントのトレースマップが作成されます。たとえば、インデックスのクエリを実行する際には、その裏で検索や LLM 呼び出しなどが行われます。

`WandbCallbackHandler` を使用すると、このトレースマップを直感的に可視化して追跡できます。イベントのペイロードをキャプチャして wandb にログ記録し、必要なメタデータ（トークン総数、プロンプト、コンテキストなど）も追跡します。

さらに、このコールバックを使用してインデックスを W&B Artifacts にアップロードおよびダウンロードし、インデックスのバージョン管理を行うこともできます。

### 1. WandbCallbackHandler のインポート

まず、`WandbCallbackHandler` をインポートし、設定します。また、W&B Project やエンティティなどの追加の [`wandb.init`](../../ref/python/init.md) パラメータを渡すこともできます。

W&B run が開始され、Weights & Biases **[API キー](https:wwww.wandb.ai/authorize)** を求められます。W&B run リンクが生成され、ログ記録を開始すると LlamaIndex クエリやデータを表示できるようになります。

```python
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager, WandbCallbackHandler

# Wandb

### 4. Weights & Biases でトレースを見る

ステップ 1 で `WandbCallbackHandler` を初期化して生成された Weights and Biases run リンクをクリックします。これにより、W&B アプリのプロジェクトワークスペースに移動し、トレーステーブルとトレースタイムラインを確認できます。

![](/images/prompts/llama_index_trace.png)

### 5. トラッキングを終了する

LLM クエリのトラッキングが完了したら、以下のように wandb プロセスを終了するのが良い習慣です:

```python
wandb_callback.finish()
```

これで完了です！ Weights & Biases を使用してインデックスにクエリをログすることができます。問題が発生した場合は、`llamaindex` タグを付けて [wandb リポジトリ](https://github.com/wandb/wandb/issues) に問題を報告してください。

### 6. [任意] インデックスデータを Weights & Biases Artifacts に保存する
Weights & Biases の [Artifacts](guides/artifacts) は、バージョン管理されたデータとモデルのストレージプロダクトです。

インデックスを Artifacts にログして必要な時に使用することで、特定のインデックス呼び出しに対してインデックスにあるデータが完全に可視化されるよう、インデックスの特定バージョンをログされたトレース出力と関連付けることができます。

```python
# `index_name` に渡される文字列があなたのアーティファクト名になります
wandb_callback.persist_index(index, index_name="my_vector_store")
```

この後、W&B runページの アーティファクトタブに移動して、アップロードされたインデックスを見ることができます。

**W&B Artifacts に保存されたインデックスの使用**

Artifacts からインデックスをロードすると [`StorageContext`](https://gpt-index.readthedocs.io/en/latest/reference/storage.html) が返されます。このストレージコンテキストを使用して、LlamaIndex の [loading functions](https://gpt-index.readthedocs.io/en/latest/reference/storage/indices_save_load.html) の機能を用いてインデックスをメモリにロードします。

```python
from llama_index import load_index_from_storage

storage_context = wandb_callback.load_storage_context(
    artifact_url="<entity/project/index_name:version>"
)
index = load_index_from_storage(storage_context, service_context=service_context)
```

**注意:** [`ComposableGraph`](https://gpt-index.readthedocs.io/en/latest/reference/query/query_engines/graph_query_engine.html) の場合、インデックスのルートIDは W&B アプリのアーティファクトのメタデータタブで確認できます。

## 次のステップ

- Tables や Runs のような既存の W&B 機能を使用して LLM アプリケーションのパフォーマンスをトラッキングできます。詳細はこのチュートリアルをご覧ください:
[Tutorial: Evaluate LLM application performance](https://github.com/wandb/examples/blob/master/colabs/prompts/prompts_evaluation.ipynb)