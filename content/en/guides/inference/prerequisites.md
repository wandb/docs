---
title: "Prerequisites"
linkTitle: "Prerequisites"
weight: 1
description: >
  Set up your environment to use W&B Inference
---

Complete these steps before using the W&B Inference service through the API or UI.

{{< alert title="Tip" >}}
Before starting, review the [usage information and limits]({{< relref "usage-limits" >}}) to understand costs and restrictions.
{{< /alert >}}

## Set up your W&B account and project

You need these items to access W&B Inference:

1. **A W&B account**  
   Sign up at [W&B](https://app.wandb.ai/login?signup=true)

2. **A W&B API key**  
   Get your API key at [https://wandb.ai/authorize](https://wandb.ai/authorize)

3. **A W&B project**  
   Create a project in your W&B account to track usage

## Set up your environment (Python)

To use the Inference API with Python, you also need to:

1. Complete the general requirements above

2. Install the required libraries:

   ```bash
   pip install openai weave
   ```

{{< alert title="Note" >}}
The `weave` library is optional but recommended. It lets you trace your LLM applications. Learn more in the [Weave Quickstart]({{< relref "../quickstart" >}}).

See [usage examples]({{< relref "examples" >}}) for code samples using W&B Inference with Weave.
{{< /alert >}}

## Next steps

After completing the prerequisites:

- Check the [API reference]({{< relref "api-reference" >}}) to learn about available endpoints
- Try the [usage examples]({{< relref "examples" >}}) to see the service in action
- Use the [UI guide]({{< relref "ui-guide" >}}) to access models through the web interface 