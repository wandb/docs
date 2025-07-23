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

### Use AI models in your app

Use [W&B Weave](https://weave-docs.wandb.ai/) to manage AI models in your code. Features include tracing, output evaluation, cost estimates, and a hosted inference service and playground for comparing different large language models (LLMs) and settings.

- [Introduction](https://weave-docs.wandb.ai/)
- [Quickstart](https://weave-docs.wandb.ai/quickstart)
- [YouTube Demo](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [Try the Playground](https://weave-docs.wandb.ai/guides/tools/playground/)
- [Try W&B Inference](https://weave-docs.wandb.ai/guides/integrations/inference)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

### Develop AI models

Use [W&B Models]({{< relref "/guides/" >}}) to manage AI model development. Features include training, fine-tuning, reporting, automating hyperparameter sweeps, and utilizing the model registry for versioning and reproducibility.

- [Introduction]({{< relref "/guides/" >}})
- [Quickstart]({{< relref "/guides/quickstart/" >}})
- [YouTube Tutorial](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [Online Course](https://wandb.ai/site/courses/101/)

</div>{{% /card %}}
{{< /cardpane >}}

{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides/core/'" style="cursor: pointer; padding-left: 20px">
<h2>Core Components</h2>

Both W&B Weave and W&B Models share common components that enable and accelerate your AI/ML engineering work. 

- [Registry]({{< relref "/guides/core/registry/" >}})
- [Artifacts]({{< relref "/guides/core/artifacts/" >}})
- [Reports]({{< relref "/guides/core/reports/" >}})
- [Automations]({{< relref "/guides/core/automations/" >}})
- [Secrets]({{< relref "/guides/core/secrets.md" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/hosting'" style="cursor: pointer;padding-left:20px;">

<h2>Platform</h2>

The Weights & Biases platform can be accessed through our SaaS offering or deployed on-premise, and it provides IAM, security, monitoring, and privacy features.

- [Deployment Options]({{< relref "/guides/hosting/hosting-options/" >}})
- [Identity and access management (IAM)]({{< relref "/guides/hosting/iam/" >}})
- [Data Security]({{< relref "/guides/hosting/data-security/" >}})
- [Privacy settings]({{< relref "/guides/hosting/privacy-settings/" >}})
- [Monitoring and Usage]({{< relref "/guides/hosting/monitoring-usage/" >}})

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
