
# LLMs のイテレーション

[**Colab ノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb)

**Weights & Biases Prompts** は、LLM を活用したアプリケーション開発のための LLMOps ツールセットです。

W&B Prompts を使うと、LLM の実行フローを可視化・調査したり、LLM の入力と出力を分析したり、中間結果を確認したり、プロンプトや LLM チェーンの設定を安全に保存・管理することができます。

## インストール

```python
!pip install "wandb==0.15.2" -qqq
!pip install "langchain==v0.0.158" openai -qqq
```

## セットアップ

このデモでは [OpenAI key](https://platform.openai.com) が必要です。

```python
import os
from getpass import getpass

if os.getenv("OPENAI_API_KEY") is None:
  os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")
```

# W&B Prompts

W&B は現在、__Trace__ というツールをサポートしています。Trace は以下の3つの主要なコンポーネントで構成されています：

**Trace table**: チェーンの入力と出力の概要。

**Trace timeline**: チェーンの実行フローを表示し、コンポーネントの種類に応じて色分けされます。

**Model architecture**: チェーンの構造と各コンポーネントの初期化に使用されたパラメータの詳細。

このセクションを実行すると、新しいパネルが自動的にワークスペースに作成され、各実行、トレース、およびモデル・アーキテクチャが表示されます。

![prompts_1](/images/tutorials/prompts_quickstart/prompts.png)

![prompts_2](/images/tutorials/prompts_quickstart/prompts2.png)

`WandbTracer` をインポートし、オプションで `wandb.init()` の引数を含む辞書を定義できます。これには、プロジェクト名やチーム名、エンティティなどが含まれます。wandb.init の詳細については、API リファレンス・ガイドを参照してください。

```python
from wandb.integration.langchain import WandbTracer

wandb_config = {"project": "wandb_prompts_quickstart"}
```

### LangChain で数値計算

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
```

```python
llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
  tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

LangChain チェーンやエージェントを呼び出すときに `WandbTracer` を渡して、Trace を W&B にログとして記録します。

```python
questions = [
    "Find the square root of 5.4.",
    "What is 3 divided by 7.34 raised to the power of pi?",
    "What is the sin of 0.47 radians, divided by the cube root of 27?",
    "what is 1 divided by zero"
]
for question in questions:
  try:
    answer = agent.run(question, callbacks=[WandbTracer(wandb_config)])
    print(answer)
  except Exception as e:
    print(e)
    pass
```

セッションを終了する際には、`WandbTracer.finish()` を呼び出して、wandb run が正常に終了するようにするのがベストプラクティスです。

```python
WandbTracer.finish()
```

# LangChain を使わない場合の実装

もし LangChain を使いたくない、特に独自のインテグレーションを書くかチームのコードに計器を設置したい場合は、それも問題ありません！ `TraceTree` と `Span` について学びましょう！

![prompts_3](/images/tutorials/prompts_quickstart/prompts3.png)

**注意:** W&B Runs は必要に応じて複数のトレースを単一の run にログすることをサポートしています。つまり、毎回新しい run を作成せずに複数回 `run.log` を呼び出すことができます。

```python
from wandb.sdk.data_types import trace_tree
import wandb
```

Span は作業単位を表し、`AGENT`, `TOOL`, `LLM` または `CHAIN` のタイプを持つことができます。

```python
parent_span = trace_tree.Span(
  name="Example Span",
  span_kind = trace_tree.SpanKind.AGENT
)
```

Span は入れ子にすることができます（そしてすべきです！）。

```python
# ツールへの呼び出しのための Span を作成
tool_span = trace_tree.Span(
  name="Tool 1", 
  span_kind = trace_tree.SpanKind.TOOL
)

# LLM チェーンへの呼び出しのための Span を作成
chain_span = trace_tree.Span(
  name="LLM CHAIN 1", 
  span_kind = trace_tree.SpanKind.CHAIN
)

# LLM チェーンによって呼び出される LLM のための Span を作成
llm_span = trace_tree.Span(
  name="LLM 1", 
  span_kind = trace_tree.SpanKind.LLM
)
chain_span.add_child_span(llm_span)
```

Span の入力と出力を追加することができます：

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
  {"input": "calculate: 2023 - 1998", "system": "you are a helpful assistant", }, 
  {"response": "25", "tokens_used":218}
)

parent_span.add_child_span(tool_span)
parent_span.add_child_span(chain_span)

parent_span.add_named_result({"user": "calculate: 2023 - 1998"},
                             {"response": "25 years old"})
```

その後、以下のように parent_span を W&B にログすることができます。

```python
run = wandb.init(name="manual_span_demo", project="wandb_prompts_demo")
run.log({"trace": trace_tree.WBTraceTree(parent_span)})
run.finish()
```

生成された W&B Run のリンクをクリックすると、作成された Trace を確認することができるワークスペースに移動します。