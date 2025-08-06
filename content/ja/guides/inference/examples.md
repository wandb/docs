---
title: 使用例
description: 実用的なコード例とともに W&B Inference の使い方を学びましょう
linkTitle: Examples
weight: 50
---

これらの例では、トレース、評価、比較のために Weave と一緒に W&B Inference を使用する方法を示します。

## 基本例: Weave で Llama 3.1 8B をトレース

この例では、**Llama 3.1 8B** モデルにプロンプトを送信し、その呼び出しを Weave でトレースする方法を紹介します。トレース機能では LLM 呼び出しの入力と出力全体を取得し、パフォーマンスを監視し、Weave UI で結果を分析できます。

{{< alert title="ヒント" >}}
[Weave でのトレース](https://weave-docs.wandb.ai/guides/tracking/tracing)の詳細をご覧ください。
{{< /alert >}}

この例では以下を行います：
- チャット補完リクエストを行う `@weave.op()`デコレーター付き関数を定義します
- トレースが自動的にあなたの W&B entity や project に記録・リンクされます
- 関数が自動的にトレースされ、入力、出力、レイテンシ、メタデータがログされます
- 結果がターミナルに表示され、**Traces** タブ（[https://wandb.ai](https://wandb.ai)）でトレースを確認できます

この例を実行する前に、[前提条件]({{< relref "prerequisites" >}})を完了してください。

```python
import weave
import openai

# トレース対象の Weave team と project を設定
weave.init("<your-team>/<your-project>")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',

    # https://wandb.ai/authorize から APIキー を取得
    api_key="<your-api-key>",

    # W&B Inference の利用状況を記録するために必要
    project="wandb/inference-demo",
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

# トレース付きで実行してログ
output = run_chat()
print(output)
```

コード実行後は、以下の方法で Weave 上のトレースを確認できます:
1. ターミナルに出力されるリンク（例: `https://wandb.ai/<your-team>/<your-project>/r/call/01977f8f-839d-7dda-b0c2-27292ef0e04g`）をクリック
2. もしくは [https://wandb.ai](https://wandb.ai) で **Traces** タブを選択

## 応用例: Weave Evaluations と Leaderboards の利用

モデル呼び出しのトレースだけでなく、パフォーマンスの評価やリーダーボードの公開も可能です。この例では、2つのモデルを質問応答データセットで比較します。

この例を実行する前に、[前提条件]({{< relref "prerequisites" >}})を完了してください。

```python
import os
import asyncio
import openai
import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

# トレース対象の Weave team と project を設定
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
            # W&B Inference の利用状況を記録するために必要
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
    description="QA データセットでモデルを比較",
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

コード実行後、あなたの W&B アカウント（[https://wandb.ai/](https://wandb.ai/)）で:

- **Traces** タブを選択して[トレースを見る](https://weave-docs.wandb.ai/guides/tracking/tracing)
- **Evals** タブを選択して[モデルの評価結果を見る](https://weave-docs.wandb.ai/guides/core-types/evaluations)
- **Leaders** タブを選択して[生成されたリーダーボードを見る](https://weave-docs.wandb.ai/guides/core-types/leaderboards)

{{< img src="/images/inference/inference-advanced-evals.png" alt="モデル評価結果を見る" >}}

{{< img src="/images/inference/inference-advanced-leaderboard.png" alt="リーダーボードを見る" >}}

## 次のステップ

- 利用可能な全メソッドの[APIリファレンス]({{< relref "api-reference" >}})を参照
- [UI]({{< relref "ui-guide" >}})でモデルを試してみましょう