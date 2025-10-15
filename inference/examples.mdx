---
title: "Usage Examples"
linkTitle: "Examples"
weight: 50
description: >
  Learn how to use W&B Inference with practical code examples
---

These examples show how to use W&B Inference with Weave for tracing, evaluation, and comparison.

## Basic example: Trace Llama 3.1 8B with Weave

This example shows how to send a prompt to the **Llama 3.1 8B** model and trace the call with Weave. Tracing captures the full input and output of the LLM call, monitors performance, and lets you analyze results in the Weave UI.

{{< alert title="Tip" >}}
Learn more about [tracing in Weave](https://weave-docs.wandb.ai/guides/tracking/tracing).
{{< /alert >}}

In this example:
- You define a `@weave.op()`-decorated function that makes a chat completion request
- Your traces are recorded and linked to your W&B entity and project
- The function is automatically traced, logging inputs, outputs, latency, and metadata
- The result prints in the terminal, and the trace appears in your **Traces** tab at [https://wandb.ai](https://wandb.ai)

Before running this example, complete the [prerequisites]({{< relref "prerequisites" >}}).

```python
import weave
import openai

# Set the Weave team and project for tracing
weave.init("<your-team>/<your-project>")

client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',

    # Get your API key from https://wandb.ai/authorize
    api_key="<your-api-key>",

    # Optional: Team and project for usage tracking
    project="wandb/inference-demo",
)

# Trace the model call in Weave
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

# Run and log the traced call
output = run_chat()
print(output)
```

After running the code, view the trace in Weave by:
1. Clicking the link printed in the terminal (for example: `https://wandb.ai/<your-team>/<your-project>/r/call/01977f8f-839d-7dda-b0c2-27292ef0e04g`)
2. Or navigating to [https://wandb.ai](https://wandb.ai) and selecting the **Traces** tab

## Advanced example: Use Weave Evaluations and Leaderboards

Besides tracing model calls, you can also evaluate performance and publish leaderboards. This example compares two models on a question-answer dataset.

Before running this example, complete the [prerequisites]({{< relref "prerequisites" >}}).

```python
import os
import asyncio
import openai
import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

# Set the Weave team and project for tracing
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
            # Get your API key from https://wandb.ai/authorize
            api_key="<your-api-key>",
            # Optional: Team and project for usage tracking
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

After running this code, go to your W&B account at [https://wandb.ai/](https://wandb.ai/) and:

- Select the **Traces** tab to [view your traces](https://weave-docs.wandb.ai/guides/tracking/tracing)
- Select the **Evals** tab to [view your model evaluations](https://weave-docs.wandb.ai/guides/core-types/evaluations)
- Select the **Leaders** tab to [view the generated leaderboard](https://weave-docs.wandb.ai/guides/core-types/leaderboards)

{{< img src="/images/inference/inference-advanced-evals.png" alt="View your model evaluations" >}}

{{< img src="/images/inference/inference-advanced-leaderboard.png" alt="View your leaderboard" >}}

## Next steps

- Explore the [API reference]({{< relref "api-reference" >}}) for all available methods
- Try models in the [UI]({{< relref "ui-guide" >}})
