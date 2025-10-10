---
title: Weights & Biases Documentation
---

<div style="padding-top:50px;">&nbsp;</div>
<div style="max-width:1600px; margin: 0 auto">
{{< banner title="Weights & Biases Documentation" background="/images/support/support_banner.png" >}}
Choose the product for which you need documentation.
{{< /banner >}}

<div class="top-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
{{< img src="/icons/Name=Models, Mode=Dark.svg" width="60" height="60" >}}
</div>
<h2>W&B Models</h2>

### Develop AI models

Use [W&B Models]({{< relref "/guides/" >}}) to manage AI model development. Features include training, fine-tuning, reporting, automating hyperparameter sweeps, and utilizing the model registry for versioning and reproducibility.

- [Introduction]({{< relref "/guides/" >}})
- [Quickstart]({{< relref "/guides/quickstart/" >}})
- [YouTube Tutorial](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [Online Course](https://wandb.ai/site/courses/101/)

</div>{{% /card %}}

{{% card %}}<div onclick="window.location.href='https://weave-docs.wandb.ai'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
{{< img src="/icons/Name=Weave, Mode=Dark.svg" width="60" height="60" >}}
</div>
<h2>W&B Weave</h2>

### Use AI models in your app

Use [W&B Weave](https://weave-docs.wandb.ai/) to manage AI models in your code. Features include tracing, output evaluation, cost estimates, and a hosted inference service and playground for comparing different large language models (LLMs) and settings.

- [Introduction](https://weave-docs.wandb.ai/)
- [Quickstart](https://weave-docs.wandb.ai/quickstart)
- [YouTube Demo](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [Try the Playground](https://weave-docs.wandb.ai/guides/tools/playground/)
- [Use Weave in your W&B runs]({{< relref "/guides/weave/set-up-weave" >}})

</div>{{% /card %}}
{{< /cardpane >}}

</div>

<div class="bottom-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides/inference/'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
{{< img src="/icons/Name=Inference, Mode=Dark.svg" width="60" height="60" >}}
</div>
<h2>W&B Inference</h2>

### Access foundation models

Use [W&B Inference]({{< relref "/guides/inference/" >}}) to access leading open-source foundation models through an OpenAI-compatible API. Features include multiple model options, usage tracking, and integration with Weave for tracing and evaluation.

- [Introduction]({{< relref "/guides/inference/" >}})
- [Available Models]({{< relref "/guides/inference/models/" >}})
- [API Reference]({{< relref "/guides/inference/api-reference/" >}})
- [Try in Playground](https://wandb.ai/inference)

</div>{{% /card %}}

{{% card %}}<div onclick="window.location.href='/guides/training/'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
{{< img src="/icons/Name=Training, Mode=Dark.svg" width="60" height="60" >}}
</div>
<h2>W&B Training</h2>

### Post-train your models

Now in public preview, use [W&B Training]({{< relref "/guides/training/" >}}) to post-train large language models using serverless reinforcement learning (RL). Features include fully managed GPU infrastructure, integration with ART and RULER, and automatic scaling for multi-turn agentic tasks.

- [Introduction]({{< relref "/guides/training/" >}})
- [Prerequisites]({{< relref "/guides/training/prerequisites/" >}})
- [Serverless RL]({{< relref "/guides/training/serverless-rl/" >}})
- [API Reference]({{< relref "/ref/training" >}})

</div>{{% /card %}}
{{< /cardpane >}}

</div>

<!-- End max-width constraing -->
</div>
<!-- HTML override just for landing page -->
<style>
.td-card-group { margin: 0 auto }
p { overflow: hidden; display: block; }
ul { margin-left: 50px; }

/* Make all cards uniform size in 2x2 grid */
.top-row-cards .td-card-group,
.bottom-row-cards .td-card-group {
    max-width: 100%;
    display: flex;
    justify-content: center;
}

.td-card {
    max-width: 480px !important;
    min-width: 480px !important;
    margin: 0.75rem !important;
    flex: 0 0 auto;
}

/* Ensure consistent height for all cards */
.td-card .card {
    height: 100%;
    min-height: 320px;
}
</style>
