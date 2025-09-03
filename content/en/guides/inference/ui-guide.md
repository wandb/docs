---
title: "UI Guide"
weight: 60
description: >
  Access W&B Inference models through the web interface
---

Learn how to use the W&B Inference service through the web UI. Before using the UI, complete the [prerequisites]({{< relref "prerequisites" >}}).

## Access the Inference service

You can access the Inference service from three places:

### Direct link

Go to [https://wandb.ai/inference](https://wandb.ai/inference).

### From the Inference tab

1. Go to your W&B account at [https://wandb.ai/](https://wandb.ai/)
2. Select **Inference** from the left sidebar
3. A page displays with available models and model information

{{< img src="/images/inference/inference-playground-single.png" alt="Using an Inference model in the Playground" >}}

### From the Playground tab

1. Select **Playground** from the left sidebar. The Playground chat UI appears
2. Hover over **W&B Inference** in the LLM dropdown list. A dropdown with available models appears on the right
3. From the models dropdown, you can:
   - Click any model name to [try it in the Playground](#try-a-model-in-the-playground)
   - [Compare multiple models](#compare-multiple-models)

{{< img src="/images/inference/inference-playground.png" alt="The Inference models dropdown in Playground" >}}

## Try a model in the Playground

After [selecting a model](#access-the-inference-service), you can test it in the Playground. Available actions include:

- [Customize model settings and parameters](https://weave-docs.wandb.ai/guides/tools/playground#customize-settings)
- [Add, retry, edit, and delete messages](https://weave-docs.wandb.ai/guides/tools/playground#message-controls)
- [Save and reuse a model with custom settings](https://weave-docs.wandb.ai/guides/tools/playground#saved-models)
- [Compare multiple models](#compare-multiple-models)

## Compare multiple models

You can compare Inference models side by side in the Playground. Access the Compare view from two places:

### From the Inference tab

1. Select **Inference** from the left sidebar. The available models page appears
2. Click anywhere on a model card (except the model name) to select it. The card border turns blue
3. Repeat for each model you want to compare
4. Click **Compare N models in the Playground** on any selected card. `N` shows the number of models selected
5. The comparison view opens

Now you can compare models and use all features from [Try a model in the Playground](#try-a-model-in-the-playground).

{{< img src="/images/inference/inference-playground-compare.png" alt="Select multiple models to compare in Playground" >}}

### From the Playground tab

1. Select **Playground** from the left sidebar. The Playground chat UI appears
2. Hover over **W&B Inference** in the LLM dropdown list. The models dropdown appears on the right
3. Select **Compare** from the dropdown. The **Inference** tab appears
4. Click anywhere on a model card (except the model name) to select it. The card border turns blue
5. Repeat for each model you want to compare
6. Click **Compare N models in the Playground** on any selected card. The comparison view opens

Now you can compare models and use all features from [Try a model in the Playground](#try-a-model-in-the-playground).

## View billing and usage information

Organization admins can track credit balance, usage history, and upcoming bills from the W&B UI:

1. Go to the W&B **Billing** page in the UI
2. Find the Inference billing information card in the bottom right corner
3. From here you can:
   - Click **View usage** to see your usage over time
   - View upcoming inference charges (for paid plans)

{{< alert title="Tip" >}}
Visit the [Inference pricing page](https://wandb.ai/site/pricing/inference) for per-model pricing details.
{{< /alert >}}

## Next steps

- Review [available models]({{< relref "models" >}}) to find the best one for your needs
- Try the [API]({{< relref "api-reference" >}}) for programmatic access
- See [usage examples]({{< relref "examples" >}}) for code samples 