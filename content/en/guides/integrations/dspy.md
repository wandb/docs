---
description: How to optimize with DSPy program with W&B.
menu:
  default:
    identifier: dspy
    parent: integrations
title: DSPy Program Optimization
weight: 500
---


### Overview

Use W&B with DSPy to track and improve your program optimization workflow. Pairing W&B with [Weave DSPy integration](https://weave-docs.wandb.ai/guides/integrations/dspy) gives you the best of both worlds:

- Metrics, and Tables in W&B
- End-to-end DSPy tracing, detailed inputs/outputs, token/cost/latency, and eval context in Weave

If you’re optimizing DSPy modules (e.g., via MIPROv2) we recommend enabling both W&B and Weave.

### Install the required library and log in

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. Set the `WANDB_API_KEY` [environment variable]({{< relref "/guides/models/track/environment-variables.md" >}}) to your API key.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. Install the `wandb` library and log in.


    ```shell
    pip install wandb weave dspy

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb weave dspy
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python" %}}

```notebook
!pip install wandb weave dspy

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

If you are using W&B for the first time you might want to check out our [quickstart]({{< relref "/guides/quickstart.md" >}})


### Track program optimization with W&B Models (experimental)

For optimizers that use `dspy.Evaluate` under the hood (for example, `MIPROv2`), you can initialize a W&B run and enable the `WandbDSPyCallback` to log evaluation metrics over time and track the evolution of the program signature as W&B Tables.

```python
import dspy
from dspy.datasets import MATH

import weave
import wandb
from wandb.integration.dspy import WandbDSPyCallback

wandb_project = "dspy-eval-models"

# Initialize Weave and W&B
weave.init(wandb_project)
wandb.init(project=wandb_project)

# Add the callback to DSPy settings
dspy.settings.callbacks.append(WandbDSPyCallback())

# Configure models and dataset
gpt4o_mini = dspy.LM('openai/gpt-4o-mini', max_tokens=2000)
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=2000, cache=True)
dspy.configure(lm=gpt4o_mini)

dataset = MATH(subset='algebra')

module = dspy.ChainOfThought("question -> answer")

THREADS = 24

optimizer_kwargs = dict(
    num_threads=THREADS, teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o_mini
)
optimizer = dspy.MIPROv2(metric=dataset.metric, auto="light", **optimizer_kwargs)

compile_kwargs = dict(
    requires_permission_to_run=False,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
)
optimized_module = optimizer.compile(module, trainset=dataset.train, **compile_kwargs)
print(optimized_module)
```

Running this program will give you both a W&B run URL and a Weave URL. In W&B view logged metrics and the W&B Table that captures your program signature over time. From the run’s Overview tab, you can also find links to the associated Weave traces for deeper inspection.

    {{< img src="/images/integrations/dspy_run_page.png" alt="Cohere fine-tuning dashboard" >}}

For more on tracing, evaluation, and optimization with DSPy in Weave, see the [Weave DSPy guide](https://weave-docs.wandb.ai/guides/integrations/dspy).
