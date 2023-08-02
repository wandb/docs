---
description: The Prompts Quickstart shows how to visualise and debug the execution flow of your LLM chains and pipelines
displayed_sidebar: ja
---

# クイックスタート

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/prompts-quickstart)


<head>
  <title>Prompts Quickstart</title>
</head>

このクイックスタートガイドでは、[Trace](intro.md)を使用してLangChainや他のLLMチェーンへの呼び出しを視覚化およびデバッグする方法について説明します。

<!-- このクイックスタートガイドでは、重みとバイアス（W&B）Promptsツールを使用して、LLMチェーンまたは開発フローの実行フローを視覚化およびデバッグする方法について説明します。 -->


## LangChainでTraceを使用する

コード1行で、W&B Traceは自動的かつ連続的に[LangChainモデル](https://python.langchain.com/en/latest/modules/models.html)、[チェーン](https://python.langchain.com/en/latest/modules/chains.html)、または[エージェント](https://python.langchain.com/en/latest/modules/agents.html)への呼び出しをログに記録します。

以下の手順に従って、LangChainを視覚化およびデバッグします。このデモでは、LangChain Mathエージェントを使用します。

### 1. WandbTracerをインポートして初期化する

まず、`wandb.integration.langchain`から`WandbTracer`をインポートします。次に、`init()`メソッドを呼び出してW&BがLangChainモデル、チェーン、またはエージェントへの呼び出しを監視するようにします。

```python
from wandb.integration.langchain import WandbTracer

WandbTracer.init({"project": "wandb_prompts"})
```

オプションで、`wandb.init()`が受け付ける引数が含まれた辞書を`WandbTracer.init`に渡すことができます。これには、プロジェクト名、チーム名、エンティティなどが含まれます。[`wandb.init`](../../ref/python/init.md)に関する詳細は、APIリファレンスガイドを参照してください。

チェーン実行が完了すると、LangChainオブジェクトへのコールはW&Bトレースに自動的にログされます。

### 2. LangChainエージェントの設定

OpenAI Langchainエージェントをインポートし、`load_tools`で数学ツール（関数）を作成します。次に、ツールオブジェクトを`initialize_agent`に渡して[`initialize_agent`](https://python.langchain.com/en/latest/_modules/langchain/agents/initialize.html)メソッドで数学エージェントを作成します。

```python
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 3. エージェントにコールを行う

エージェント（この例では`math_agent`）に対して行ったすべてのコールは、実行が完了するとログされます。

オブジェクトの作成に使用されたパラメータもログに記録されます。

```python
questions = [
    "5.4の平方根を求めてください。",
    "3を7.34で割った数をπのべき乗にしてください。",
    "0.47ラジアンのsinを27の立方根で割ってください。"
]

for question in questions:
  try:
    answer = math_agent.run(question)
    print(answer)
  except Exception as e:
    print(e)
    pass
```

### 4. トレースを表示する

前のステップで`WandbTracer.init` によって生成された Weights & Biases [run](../runs/intro.md) リンクをクリックしてください。これにより、W&Bアプリ内のプロジェクトワークスペースにリダイレクトされます。

作成したrunを選択して、LLMのトレーステーブル、トレースタイムライン、およびアーキテクチャーを表示します。

![](/images/tracer/trace_timeline_detailed.png)

### 5. 監視を停止する
開発が終わったら、`WandbTracer.finish`を呼び出してすべてのW&Bプロセスを閉じることをお勧めします。

```python
WandbTracer.finish()
```
## 任意のLLMチェーンやプラグインとTraceを使用する

Traceでログを取る際、1つのrunに複数のLLM、ツール、チェーン、エージェントがログされるため、モデルからの各生成ごとに新しいrunを開始する必要はありません。各呼び出しがTraceテーブルに追加されます。

Traceを独自のチェーン、プラグイン、開発フローと使用するには、まず`Span`および`TraceTree`データ型を使用してTraceを作成する必要があります。 _Span_は作業の単位を表します。

### 1. Spanの作成
まず、Spanオブジェクトを作成します。`wandb.sdk.data_types`から`trace_tree`をインポートします。

```python
from wandb.sdk.data_types import trace_tree

# span = trace_tree.Span(name="Example Span")
# 親Span - 高レベルエージェントのためのSpanを作成します
agent_span = trace_tree.Span(name="Auto-GPT", span_kind = trace_tree.SpanKind.AGENT)
```

Spanは、`AGENT`、`CHAIN`、`TOOL`、`LLM`のタイプがあります。

### 2. 子Spanの追加
親Span内に子Spanをネストさせて、Traceのタイムラインビューで正しい順序でネストされるようにします。以下では、2つの子Spanと1つの孫Spanが作成されます。

```python
tool_span = trace_tree.Span(
  name="Tool 1", span_kind = trace_tree.SpanKind.TOOL
)

chain_span = trace_tree.Span(
  name="LLM CHAIN 1", span_kind = trace_tree.SpanKind.CHAIN
)

llm_span = trace_tree.Span(
  name="LLM 1", span_kind = trace_tree.SpanKind.LLM
)

chain_span.add_child_span(llm_span)
agent_span.add_child_span(tool_span)
agent_span.add_child_span(chain_span)
```

### 3. 入力と出力を追加する

スパンに入力と出力データを追加する

```python
tool_span.add_named_result(
  {"input": "search: google founded in year"}, 
  {"response": "1998"}
)

chain_span.add_named_result(
  {"input": "calculate: 2023 - 1998"}, 
  {"response": "25"}
)

llm_span.add_named_result(
  {"system": "you are a helpful assistant", 
    "input": "calculate: 2023 - 1998"}, 
  {"response": "25", "tokens_used": 218}
)

agent_span.add_named_result(

  {"user": "Googleは何歳ですか？"},

  {"response": "25歳"}

)

```

### 4. スパンをWeights & BiasesのTraceにログ

これにより、Trace Table、Trace Timeline、およびモデルアーキテクチャを視覚化できます。

```python

import wandb 



trace = trace_tree.WBTraceTree(agent_span)

run = wandb.init(project="wandb_prompts")

run.log({"trace": trace})

run.finish()

```

### 5. トレースを表示

生成されたW&Bのrunリンクをクリックして、LLMのトレースを確認します。