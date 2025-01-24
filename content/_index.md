---
title: Weights & Biases Documentation
---
<div style="padding-top:50px;">&nbsp;</div>
<div style="max-width:1200px; margin: 0 auto">
{{< banner title="Weights & Biases Documentation" background="/images/support/support_banner.png" >}}
Choose the product for which you need documentation.
{{< /banner >}}

{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='https://weave-docs.wandb.ai'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/weave-logo.svg" alt="W&B Weave logo" width="50" height="50"/>
</div>
<h2>W&B Weave</h2>

##### Use AI models in your app

Use [W&B Weave](https://weave-docs.wandb.ai/) to manage all aspects of integrating AI models into your code, including tracing, output evaluation, cost estimates, and using our LLM playground to help compare the various LLM models and their configuration paramaters.

- [Introduction](https://weave-docs.wandb.ai/)
- [Quickstart](https://weave-docs.wandb.ai/quickstart)
- [YouTube Demo](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [Try the Playground](https://wandb.ai/wandb/weave-playground/weave/playground) (Free [sign up](https://wandb.ai/signup) required)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

##### Build AI models

Use [W&B Models]({{< relref "/guides/" >}}) to manage all aspects of building your own AI models, including training, fine-tuning, reporting, automating hyperparameter sweeps, and using our model registry to assist with versioning and reproducibility.

- [Introduction]({{< relref "/guides/" >}})
- [Quickstart]({{< relref "/guides/quickstart/" >}})
- [YouTube Tutorial](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [Online Course](https://www.wandb.courses/courses/wandb-101)

</div>{{% /card %}}
{{< /cardpane >}}

<!-- End max-width constraing -->
</div>
<!-- HTML override just for landing page -->
<style>
.td-card-group { margin: 0 auto }
p { overflow: hidden; display: block; }
ul { margin-left: 50px; }
</style>