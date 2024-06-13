
# LLMの反復

[**こちらのColabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb)

**Weights & Biases Prompts** は、LLMを利用したアプリケーションの開発のために設計されたLLMOpsツールのスイートです。

W&B Promptsを使用すると、LLMの実行フローの視覚化と検査、入力と出力の分析、中間結果の表示、およびプロンプトとLLMチェーン設定の安全な保存と管理が可能です。

## インストール

```python
!pip install "wandb==0.15.2" -qqq
!pip install "langchain==v0.0.158" openai -qqq
```

## 設定

このデモには[OpenAI key](https://platform.openai.com)が必要です

```python
import os
from getpass import getpass

if os.getenv("OPENAI_API_KEY") is None:
  os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")
```

# W&B Prompts

W&Bは現在、__Trace__ と呼ばれるツールをサポートしています。Traceは以下の3つの主要コンポーネントから構成されています：

**Trace table**: チェーンの入力と出力の概要。

**Trace timeline**: チェーンの実行フローを表示し、コンポーネントの種類ごとに色分けされています。

**Model architecture**: チェーンの構造および各コンポーネントを初期化するために使用されるパラメーターに関する詳細を表示します。

このセクションを実行すると、ワークスペースに各実行、トレース、およびモデルのアーキテクチャが表示された新しいパネルが自動的に作成されます。

![prompts_1](/images/tutorials/prompts_quickstart/prompts.png)

![prompts_2](/images/tutorials/prompts_quickstart/prompts2.png)

`WandbTracer`をインポートし、後に`WandbTracer`に渡される`wandb.init()`の引数を含む辞書をオプションで定義します。これにはプロジェクト名、チーム名、エンティティなどが含まれます。`wandb.init`についての詳細は、APIリファレンスガイドを参照してください。

```python
from wandb.integration.langchain import WandbTracer

wandb_config = {"project": "wandb_prompts_quickstart"}
```

### LangChainを使った数式

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

LangChainチェーンやエージェントを呼び出す際に`WandbTracer`を渡して、W&Bにトレースをログします。

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

セッションを終了するときは、`WandbTracer.finish()`を呼び出してwandb runが適切に終了するようにすることが最善の方法です。

```python
WandbTracer.finish()
```

# Non-Lang Chainの実装

もしLangchainを使用したくない場合、特にチームのコードに統合やインストゥルメントを行いたい場合はどうすればよいでしょうか？それも完全に問題ありません！`TraceTree`と`Span`について学びましょう！

![prompts_3](/images/tutorials/prompts_quickstart/prompts3.png)

**注:** W&B Runsは必要なだけ多くのトレースを単一のrunにログすることをサポートしています。つまり、毎回新しいrunを作成する必要はなく、`run.log`を複数回呼び出すことができます。

```python
from wandb.sdk.data_types import trace_tree
import wandb
```

Spanは作業単位を表し、Spanは`AGENT`、`TOOL`、`LLM`、または`CHAIN`のタイプを持つことができます。

```python
parent_span = trace_tree.Span(
  name="Example Span", 
  span_kind = trace_tree.SpanKind.AGENT
)
```

Spanはネストすることができます（そしてそうするべきです！）。

```python
# ツールの呼び出しのためのSpanを作成
tool_span = trace_tree.Span(
  name="Tool 1", 
  span_kind = trace_tree.SpanKind.TOOL
)

# LLMチェーンの呼び出しのためのSpanを作成
chain_span = trace_tree.Span(
  name="LLM CHAIN 1", 
  span_kind = trace_tree.SpanKind.CHAIN
)

# LLMチェーンによって呼び出されるLLMの呼び出しのためのSpanを作成
llm_span = trace_tree.Span(
  name="LLM 1", 
  span_kind = trace_tree.SpanKind.LLM
)
chain_span.add_child_span(llm_span)
```

Spanの入力と出力は次のように追加できます。

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

次のように親SpanをW&Bにログすることができます。

```python
run = wandb.init(name="manual_span_demo", project="wandb_prompts_demo")
run.log({"trace": trace_tree.WBTraceTree(parent_span)})
run.finish()
```

生成されたW&B Runのリンクをクリックすると、作成されたトレースを検査するためのワークスペースに移動できます。