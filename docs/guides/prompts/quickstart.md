---
description: プロンプトクイックスタートでは、LLMチェーンとパイプラインの実行フローを視覚化し、デバッグする方法を示しています
displayed_sidebar: default
---

# Prompts クイックスタート

[**Colab ノートブックで試す →**](http://wandb.me/prompts-quickstart)

<head>
  <title>Prompts クイックスタート</title>
</head>

このクイックスタートガイドでは、[Trace](intro.md) を使用して LangChain、LlamaIndex または独自の LLM Chain や Pipeline への呼び出しを可視化およびデバッグする方法を説明します:

1. **[Langchain:](#use-wb-trace-with-langchain)** 自動ログのための 1 行の LangChain 環境変数またはコンテキストマネージャーインテグレーションを使用。

2. **[LlamaIndex:](#use-wb-trace-with-llamaindex)** 自動ログのための LlamaIndex の W&B コールバックを使用。

3. **[カスタム使用](#use-wb-trace-with-any-llm-pipeline-or-plug-in)**: 独自のカスタムチェーンおよび LLM パイプラインコードで Trace を使用。


## W&B Trace を LangChain で使用する

:::info
**バージョン** `wandb >= 0.15.4` および `langchain >= 0.0.218` を使用してください。
:::

LangChain の 1 行の環境変数を使用することで、W&B Trace は LangChain モデル、チェーン、エージェントへの呼び出しを継続的にログします。

W&B Trace の文書については、[LangChain の文書](https://python.langchain.com/docs/integrations/providers/wandb_tracing) でも確認できます。

このクイックスタートでは、LangChain Math エージェントを使用します:

### 1. LANGCHAIN_WANDB_TRACING 環境変数を設定する

まず、LANGCHAIN_WANDB_TRACING 環境変数を true に設定します。これにより、LangChain で Weights & Biases の自動ログが有効になります:

```python
import os

# langchain のために wandb ログを有効化
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
```

以上です！ これで LangChain LLM、チェーン、ツール、エージェントへのすべての呼び出しが Weights & Biases にログされます。

### 2. Weights & Biases の設定を構成する
追加の Weights & Biases の [環境変数](/guides/track/environment-variables) を設定して、通常 `wandb.init()` に渡されるパラメータを設定することもできます。`WANDB_PROJECT` や `WANDB_ENTITY` など、どこにログが送られるかをより細かく制御するためのパラメータを設定します。[`wandb.init`](../../ref/python/init.md) についての詳細は、API リファレンスガイドを参照してください。

```python
# オプションで wandb の設定や構成を設定
os.environ["WANDB_PROJECT"] = "langchain-tracing"
```


### 3. LangChain エージェントを作成する
LangChain を使用して標準的な Math エージェントを作成します:

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```


### 4. エージェントを実行し、Weights & Biases ログを開始する
エージェントを呼び出して通常通り LangChain を使用します。Weights & Biases の run が開始され、Weights & Biases の **[APIキー](https:wwww.wandb.ai/authorize)** を求められます。APIキーを入力すると、エージェントの呼び出しの入力と出力が Weights & Biases アプリにストリームされ始めます。

```python
# いくつかの数学の質問例
questions = [
    "5.4 の平方根を求めなさい。",
    "3 を 7.34 の pi 乗で割ると何になりますか？",
    "0.47 ラジアンの sin を求め、それを 27 の立方根で割ります。",
]

for question in questions:
    try:
        # 通常通りエージェントを呼び出す
        answer = math_agent.run(question)
        print(answer)
    except Exception as e:
        # Weights & Biases にエラーもログされます
        print(e)
        pass
```

それぞれのエージェントの実行が完了すると、LangChain オブジェクト内のすべての呼び出しが Weights & Biases にログされます。


### 5. Weights & Biases でトレースを表示する

前のステップで生成された W&B [run](../runs/intro.md) リンクをクリックします。これで W&B アプリのプロジェクトワークスペースにリダイレクトされます。

作成した run を選択して、トレーステーブル、トレースタイムライン、および LLM のモデルアーキテクチャを表示します。

![](/images/prompts/trace_timeline_detailed.png)


### 6. LangChain コンテキストマネージャー
ユースケースに応じて、コンテキストマネージャーを使用して W&B へのログを管理する方が適している場合があります:

```python
from langchain.callbacks import wandb_tracing_enabled

# 環境変数を解除し、代わりにコンテキストマネージャーを使用
if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

# コンテキストマネージャーを使用してトレースを有効化
with wandb_tracing_enabled():
    math_agent.run("5 を 0.123243 乗した結果は？")  # これがトレースされます

math_agent.run("2 を 0.123243 乗した結果は？")  # これはトレースされません
```

この LangChain のインテグレーションに関する問題は、タグ `langchain` をつけて [wandb レポジトリ](https://github.com/wandb/wandb/issues) に報告してください。


## 任意の LLM パイプラインまたはプラグインで W&B Trace を使用

:::info
**バージョン** `wandb >= 0.15.4` を使用してください。
:::

W&B Trace は 1 つ以上の「スパン」をログすることによって作成されます。ルートスパンが期待され、その中にネストされた子スパンを受け入れることができます。スパンは、`AGENT`, `CHAIN`, `TOOL` または `LLM` のタイプであることができます。

Trace を使用してログを記録する場合、1 つの W&B run に対して LLM、ツール、チェーン、エージェントの複数の呼び出しがログされることがあります。モデルやパイプラインの各生成後に新しい W&B run を開始する必要はありません。各呼び出しが Trace Table に追加されるだけです。

このクイックスタートでは、OpenAI モデルへの単一の呼び出しを W&B Trace に単一のスパンとしてログする方法を説明します。その後、より複雑な一連のネストされたスパンをログする方法を示します。

### 1. Trace をインポートし、Weights & Biases run を開始

`wandb.init` を呼び出して W&B run を開始します。ここで W&B プロジェクト名やエンティティ名 (W&B Team にログする場合) などを渡すことができます。完全な引数リストについては [`wandb.init`](../../ref/python/init.md) を参照してください。

W&B run を開始すると、Weights & Biases の **[APIキー](https:wwww.wandb.ai/authorize)** でログインするよう求められます。

```python
import wandb

# wandb run を開始してログを記録
wandb.init(project="trace-example")
```

W&B Team にログする場合は、`wandb.init` で `entity` 引数を設定することもできます。

### 2. Trace にログを記録する
次に、OpenAI への照会結果を W&B Trace にログします。入力および出力、開始および終了時間、OpenAI 呼び出しの成功可否、トークン使用量、および追加のメタデータをログします。

Trace クラスへの引数の完全な説明については [こちら](https://github.com/wandb/wandb/blob/653015a014281f45770aaf43627f64d9c4f04a32/wandb/sdk/data_types/trace_tree.py#L166) を参照してください。

```python
import openai
import datetime
from wandb.sdk.data_types.trace_tree import Trace

openai.api_key = "<YOUR_OPENAI_API_KEY>"

# 構成を定義
model_name = "gpt-3.5-turbo"
temperature = 0.7
system_message = "あなたは 3つの簡潔な箇条書きで Markdown を用いて必ず返答する親切なアシスタントです。"

queries_ls = [
    "フランスの首都はどこですか？",
    "卵を茹でるにはどうすれば良いですか？" * 10000,  # 意図的に openai エラーを引き起こす
    "宇宙人が到着した場合はどうすれば良いですか？",
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
        )  # ミリ秒でログ
        status = "success"
        status_message = (None,)
        response_text = response["choices"][0]["message"]["content"]
        token_usage = response["usage"].to_dict()

    except Exception as e:
        end_time_ms = round(
            datetime.datetime.now().timestamp() * 1000
        )  # ミリ秒でログ
        status = "error"
        status_message = str(e)
        response_text = ""
        token_usage = {}

    # wandb にスパンを作成
    root_span = Trace(
        name="root_span",
        kind="llm",  # 種類には "llm", "chain", "agent", "tool" が使用可能
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

    # スパンを wandb にログ
    root_span.log(name="openai_trace")
```

### 3. Weights & Biases でトレースを表示する

ステップ 2 で生成された W&B [run](../runs/intro.md) リンクをクリックします。ここで LLM のトレーステーブルおよびトレースタイムラインを表示できます。


### 4. ネストされたスパンを使用して LLM パイプラインをログ
この例では、エージェントが呼び出され、次に LLM Chain が呼び出され、OpenAI LLM が呼び出され、その後 Calculaotr ツールが「呼ばれる」エージェントをシミュレートします。

実行の各ステップの入力、出力、およびメタデータは、それぞれ独自のスパンにログされます。スパンは子を持つことができます。

```python
import time

# エージェントが回答する必要があるクエリ
query = "次の米国選挙まで何日ですか？"

# パート 1 - エージェントが開始されます...
start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

root_span = Trace(
    name="MyAgent",
    kind="agent",
    start_time_ms=start_time_ms,
    metadata={"user": "optimus_12"},
)


# パート 2 - エージェントが LLMChain に呼び出します…
chain_span = Trace(name="LLMChain", kind="chain", start_time_ms=start_time_ms)

# ルートの子としてチェーンスパンを追加
root_span.add_child(chain_span)


# パート 3 - LLMChain が OpenAI LLM に呼び出します…
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

# チェーンスパンの子として LLM スパンを追加
chain_span.add_child(llm_span)

# チェーンスパンの終了時間を更新
chain_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"response": response_text}
)

# チェーンスパンの終了時間を更新
chain_span._span.end_time_ms = llm_end_time_ms


# パート 4 - エージェントがツールに呼び出します…
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

# ルートの子としてツールスパンを追加
root_span.add_child(tool_span)


# パート 5 - ツールからの最終結果を追加
root_span.add_inputs_and_outputs(
    inputs={"query": query}, outputs={"result": days_to_election}
)
root_span._span.end_time_ms = tool_end_time_ms


# パート 6 - ルートスパンをログすることで、すべてのスパンを W&B にログ
root_span.log(name="openai_trace")
```

スパンをログすると、W&B アプリで Trace テーブルが更新されるのを確認できます。


## W&B Trace を LlamaIndex で使用

:::info
**バージョン** `wandb >= 0.15.4` および `llama-index >= 0.6.35` を使用してください。
:::

最下位レベルでは、LlamaIndex はログを追跡するための開始/終了イベントの概念 ([`CBEventTypes`](https://gpt-index.readthedocs.io/en/latest/reference/callbacks.html#llama_index.callbacks.CBEventType)) を使用します。各イベントには、LLM によって生成されたクエリおよび応答や、N チャンク作成に使用された文書の数などの情報を提供するペイロードがあります。

より高いレベルでは、最近、「トレース マップ」を構築するコールバック トレースの概念が導入されました。例えば、インデックス上でクエリを実行する場合、内部では取得、LLM の呼び出しなどが行われます。

`WandbCallbackHandler` は、このトレースマップを直感的に視覚化および追跡する方法を提供します。イベントのペイロードをキャプチャして wandb にログし、総トークン数、プロンプト、コンテキストなどの必要なメタデータも追跡します。

さらに、このコールバックを使用して、インデックスを W&B Artifacts にアップロードおよびダウンロードしてインデックスをバージョン管理することもできます。

### 1. WandbCallbackHandler のインポート

まず、`WandbCallbackHandler` をインポートし、セットアップします。追加のパラメータ [`wandb.init`](../../ref/python/init.md) も渡すことができます。例えば、W&B プロジェクトやエンティティ。

W&B run が開始され、Weights & Biases の **[APIキー](https:wwww.wandb.ai/authorize)** を求められます。W&B run のリンクが生成され、ログを記録すると、ここで LlamaIndex クエリやデータを確認できます。

```python
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager, WandbCallbackHandler

# WandbCallbackHandler を初期化し、任意の wandb.init 引数を渡す
wandb_args = {"project": "llamaindex"}
wandb_callback = WandbCallbackHandler(run_args=wandb_args)

# wandb_callback をサービスコンテキストに渡す
callback_manager = CallbackManager([wandb_callback])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
```

### 2. インデックスを構築する

テキストファイルを使用してシンプルなインデックスを構築します。

```python
docs = SimpleDirectoryReader("path_to_dir").load_data()
index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
```

### 3. インデックスにクエリを実行し、Weights & Biases ログを開始する

ロードされたインデックスを使用して、文書に対してクエリを開始します。インデックスへのすべての呼び出しは自動的に Weights & Biases にログされます。

```python
questions = [
    "著者は育ったときに何をしましたか？",
    "著者はどこか旅行しましたか？",
    "著者が愛していることは何ですか？",
]

query_engine = index.as_query_engine()

for q in questions:
    response = query_engine.query(q)
```

### 4. Weights & Biases でトレースを表示する

初期化時に生成された `WandbCallbackHandler` の Weights & Biases run リンクをクリックします。これにより、W&B アプリのプロジェクトワークスペースに移動し、トレーステーブルおよびトレースタイムラインを確認できます。

![](/images/prompts/llama_index_trace.png)

### 5. トラッキングを終了する

LLM クエリのトラッキングが完了したら、以下のように wandb プロセスを終了することをお勧めします:

```python
wandb_callback.finish()
```

それだけです！ これで Weights & Biases を使用してインデックスへのクエリをログできます。何か問題が発生した場合は、タグ `llamaindex` を付けて [wandb レポジトリ](https://github.com/wandb/wandb/issues) に問題を報告してください。

### 6. [オプション] インデックスデータを Weights & Biases Artifacts に保存する
Weights & Biases の [Artifacts](guides/artifacts) はバージョン管理されたデータおよびモデルストレージ製品です。  

アーティファクトにインデックスをログし、必要時に使用することで特定のインデックスバージョンをトレース出力と関連付け、特定のインデックス呼び出しに対するデータの完全な可視性を確保できます。

```python
# アーティファクト名として渡される文字列は `index_name` になります
wandb_callback.persist_index(index, index_name="my_vector_store")
```

その後、W&B run ページのアーティファクトタブにアクセスして、アップロードされたインデックスを確認できます。

**W&B Artifacts に保存されたインデックスを利用する方法**

アーティファクトからインデックスをロードすると [`StorageContext`](https://gpt-index.readthedocs.io/en/latest/reference/storage.html) が返されます。このストレージコンテキストを使用して、LlamaIndex [読み込み関数](https://gpt-index.readthedocs.io/en/latest/reference/storage/indices_save_load.html) の関数を使用してメモリにインデックスをロードします。

```python
from llama_index import load_index_from_storage

storage_context = wandb_callback.load_storage_context(
    artifact_url="<entity/project/index_name:version>"
)
index = load_index_from_storage(storage_context, service_context=service_context)
```

**注:** [`ComposableGraph`](https://gpt-index.readthedocs.io/en/latest/reference/query/query_engines/graph_query_engine.html) のルート ID は、W&B アプリのアーティファクトのメタデータタブで確認できます。

## 次のステップ

