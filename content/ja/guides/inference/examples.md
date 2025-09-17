---
title: 使用例
description: 実用的な コード例 を用いて W&B Inference の使い方を学びましょう
linkTitle: Examples
menu:
  default:
    identifier: ja-guides-inference-examples
weight: 50
---

これらの例では、Weave と組み合わせて W&B Inference を使い、トレース、評価、比較を行う方法を示します。

## 基本例: Weave で Llama 3.1 8B をトレース

この例では、**Llama 3.1 8B** モデルにプロンプトを送信し、Weave で呼び出しをトレースする方法を示します。トレースは LLM 呼び出しの入力と出力を完全に記録し、パフォーマンスを監視し、Weave の UI で結果を分析できるようにします。

{{< alert title="ヒント" >}}
[tracing in Weave](https://weave-docs.wandb.ai/guides/tracking/tracing) について詳しく学べます。
{{< /alert >}}

この例では:
- チャット補完リクエストを行う `@weave.op()` で装飾された関数を定義します
- トレースは記録され、あなたの W&B Entity と Project にリンクされます
- 関数は自動的にトレースされ、入力、出力、レイテンシ、メタデータをログに記録します
- 結果はターミナルに出力され、トレースは [https://wandb.ai](https://wandb.ai) の **Traces** タブに表示されます

この例を実行する前に、[prerequisites]({{< relref path="prerequisites" lang="ja" >}}) を完了してください。

```python
import weave
import openai

# トレース用の Weave の Team と Project を設定
weave.init("<your-team>/<your-project>")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',

    # https://wandb.ai/authorize から API キー を取得
    api_key="<your-api-key>",

    # W&B Inference の使用状況トラッキングに必須
    project="wandb/inference-demo",
)

# Weave でモデル呼び出しをトレース
@weave.op()
def run_chat():
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."}
        ],
    )
    return response.choices[0].message.content

# トレースした呼び出しを実行してログを記録
output = run_chat()
print(output)
```

コードを実行したら、次の方法で Weave でトレースを確認できます:
1. ターミナルに出力されたリンクをクリックする (例: `https://wandb.ai/<your-team>/<your-project>/r/call/01977f8f-839d-7dda-b0c2-27292ef0e04g`)
2. または [https://wandb.ai](https://wandb.ai) に移動して **Traces** タブを選択する

## 上級例: Weave の Evaluations と Leaderboards を使う

モデル呼び出しのトレースに加えて、性能を評価してリーダーボードを公開することもできます。この例では、質問応答のデータセットで 2 つのモデルを比較します。

この例を実行する前に、[prerequisites]({{< relref path="prerequisites" lang="ja" >}}) を完了してください。

```python
import os
import asyncio
import openai
import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

# トレース用の Weave の Team と Project を設定
weave.init("<your-team>/<your-project>")

dataset = [
    {"input": "What is 2 + 2?", "target": "4"},
    {"input": "Name a primary color.", "target": "red"},
]

@weave.op
def exact_match(target: str, output: str) -> float:
    return float(target.strip().lower() == output.strip().lower())

class WBInferenceModel(weave.Model):
    model: str

    @weave.op
    def predict(self, prompt: str) -> str:
        client = openai.OpenAI(
            base_url="https://api.inference.wandb.ai/v1",
            # https://wandb.ai/authorize から API キー を取得
            api_key="<your-api-key>",
            # W&B Inference の使用状況トラッキングに必須
            project="<your-team>/<your-project>",
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

llama = WBInferenceModel(model="meta-llama/Llama-3.1-8B-Instruct")
deepseek = WBInferenceModel(model="deepseek-ai/DeepSeek-V3-0324")

def preprocess_model_input(example):
    return {"prompt": example["input"]}

evaluation = weave.Evaluation(
    name="QA",
    dataset=dataset,
    scorers=[exact_match],
    preprocess_model_input=preprocess_model_input,
)

async def run_eval():
    await evaluation.evaluate(llama)
    await evaluation.evaluate(deepseek)

asyncio.run(run_eval())

spec = leaderboard.Leaderboard(
    name="Inference Leaderboard",
    description="Compare models on a QA dataset",
    columns=[
        leaderboard.LeaderboardColumn(
            evaluation_object_ref=get_ref(evaluation).uri(),
            scorer_name="exact_match",
            summary_metric_path="mean",
        )
    ],
)

weave.publish(spec)
```

このコードを実行したら、[https://wandb.ai/](https://wandb.ai/) のあなたの W&B アカウントで次を行います:

- **Traces** タブを選択して、[トレースを表示](https://weave-docs.wandb.ai/guides/tracking/tracing)
- **Evals** タブを選択して、[モデルの評価を表示](https://weave-docs.wandb.ai/guides/core-types/evaluations)
- **Leaders** タブを選択して、[生成されたリーダーボードを表示](https://weave-docs.wandb.ai/guides/core-types/leaderboards)

{{< img src="/images/inference/inference-advanced-evals.png" alt="モデルの評価を表示" >}}

{{< img src="/images/inference/inference-advanced-leaderboard.png" alt="リーダーボードを表示" >}}

## 次のステップ

- すべての利用可能なメソッドについては、[API reference]({{< relref path="api-reference" lang="ja" >}}) を参照してください
- [UI]({{< relref path="ui-guide" lang="ja" >}}) でモデルを試してみましょう