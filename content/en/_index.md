---
title: Weights & Biases Documentation
---
<div id="cardHolders" style="max-width:1200px; padding-top: 80px !important; padding-left: 5px;">

{{< banner title="Weights & Biases Documentation" background="/images/support/support_banner.png" >}}
W&B is the AI developer platform to build AI agents, applications,
and models with confidence.
{{< /banner >}}

{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='https://weave-docs.wandb.ai'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/weave-logo.svg" alt="W&B Weave logo" width="50" height="50"/>
</div>
<h2>W&B Weave</h2>

#### Build agentic AI applications

Use [W&B Weave](https://weave-docs.wandb.ai/) to manage AI models in your code. Features include tracing, output evaluation, cost estimates, and a playground for comparing different large language models (LLMs) and settings.

- [Introduction](https://weave-docs.wandb.ai/)
- [Quickstart](https://weave-docs.wandb.ai/quickstart)
- [YouTube Demo](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [Try the Playground](https://weave-docs.wandb.ai/guides/tools/playground/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

#### Build AI models

Use [W&B Models]({{< relref "/guides/" >}}) to manage AI model development. Features include training, fine-tuning, reporting, automating hyperparameter sweeps, and utilizing the model registry for versioning and reproducibility.

- [Introduction]({{< relref "/guides/" >}})
- [Quickstart]({{< relref "/guides/quickstart/" >}})
- [YouTube Tutorial](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [Online Course](https://wandb.ai/site/courses/101/)

</div>{{% /card %}}
{{< /cardpane >}}

<!-- End max-width constraing -->
</div>
<!-- HTML override just for landing page -->
<style>
#cardHolders div.card { min-width: 47% !important; max-width: 380px !important; margin-bottom: 25px }
#cardHolders div.banner { min-width: 47%; max-width: 930px; }
p { overflow: hidden; display: block; }
ul { margin-left: 50px; }
aside.td-sidebar { display: none; }
h2, h3, h4, h5, h6 { font-weight: 700 !important; }
</style>
