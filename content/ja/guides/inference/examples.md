---
title: 使用例
description: 実用的なコード例を使って W&B Inference の使い方を学びましょう
linkTitle: 使用例
menu:
  default:
    identifier: ja-guides-inference-examples
weight: 50
---

これらの例では、W&B Inference を Weave と組み合わせて、トレース、評価、モデル比較を行う方法を紹介します。

## 基本例: Weave で Llama 3.1 8B をトレース

この例では、**Llama 3.1 8B** モデルにプロンプトを送り、その呼び出しを Weave でトレースする方法を示します。トレースでは LLM 呼び出しの入力・出力がすべて記録され、パフォーマンスの監視や Weave UI での結果分析が可能になります。

{{< alert title="ヒント" >}}
[Weave によるトレース](https://weave-docs.wandb.ai/guides/tracking/tracing)の詳細はこちらをご覧ください。
{{< /alert >}}

この例で行うこと:
- `@weave.op()` で装飾された関数を定義し、チャット補完リクエストを作成
- トレースが W&B の entity や project に紐づいて記録される
- 関数は自動的にトレースされ、入力・出力・レイテンシ・メタデータがログされる
- 結果がターミナルに出力され、**Traces** タブ（[https://wandb.ai](https://wandb.ai)）でトレースを確認可能

この例を実行する前に、[前提条件]({{< relref path="prerequisites" lang="ja" >}}) を完了してください。

```python
import weave
import openai

# トレース用に Weave の team と project をセット
weave.init("<your-team>/<your-project>")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',

    # https://wandb.ai/authorize から APIキー を取得
    api_key="<your-api-key>",
)

# モデル呼び出しを Weave でトレース
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

# トレース呼び出しを実行してログ
output = run_chat()
print(output)
```

コードを実行した後、Weave でトレースを確認する方法:
1. ターミナルに表示されるリンクをクリック（例: `https://wandb.ai/<your-team>/<your-project>/r/call/01977f8f-839d-7dda-b0c2-27292ef0e04g`）
2. または [https://wandb.ai](https://wandb.ai) にアクセスし、**Traces** タブを選択

## 応用例: Weave Evaluations と Leaderboards の活用

モデル呼び出しのトレースだけでなく、パフォーマンス評価やリーダーボードの公開も可能です。  
この例では、2つのモデルを QA データセットで比較し、クライアント初期化時にカスタム project 名を設定して、ログの保存先を指定しています。

この例を実行する前に、[前提条件]({{< relref path="prerequisites" lang="ja" >}}) を完了してください。

```python
import os
import asyncio
import openai
import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

# トレース用に Weave の team と project をセット
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
            # https://wandb.ai/authorize から APIキー を取得
            api_key="<your-api-key>",
            # （任意）ログの保存先をカスタマイズ
            project="<your-team>/<your-project>"
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

このコードを実行した後、W&B アカウント（[https://wandb.ai/](https://wandb.ai/)）で以下を行ってください:

- **Traces** タブを選択し、[トレースを確認](https://weave-docs.wandb.ai/guides/tracking/tracing)
- **Evals** タブを選択し、[モデルの評価を確認](https://weave-docs.wandb.ai/guides/core-types/evaluations)
- **Leaders** タブを選択し、[生成されたリーダーボードを確認](https://weave-docs.wandb.ai/guides/core-types/leaderboards)

{{< img src="/images/inference/inference-advanced-evals.png" alt="モデル評価の確認" >}}

{{< img src="/images/inference/inference-advanced-leaderboard.png" alt="リーダーボードの確認" >}}

## 次のステップ

- すべての利用可能なメソッドは [APIリファレンス]({{< relref path="api-reference" lang="ja" >}}) を参照してください
- [UI]({{< relref path="ui-guide" lang="ja" >}}) でモデルを試してみましょう