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

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

### Develop AI models

Use [W&B Models]({{< relref path="/guides/" lang="ko" >}}) to manage AI model development. Features include training, fine-tuning, reporting, automating hyperparameter sweeps, and utilizing the model registry for versioning and reproducibility.

- [Introduction]({{< relref path="/guides/" lang="ko" >}})
- [Quickstart]({{< relref path="/guides/quickstart/" lang="ko" >}})
- [YouTube Tutorial](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [Online Course](https://wandb.ai/site/courses/101/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/inference/'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Inference logo" width="40" height="40"/>
</div>
<h2>W&B Inference</h2>

### Access foundation models

Use [W&B Inference]({{< relref path="/guides/inference/" lang="ko" >}}) to access leading open-source foundation models through an OpenAI-compatible API. Features include multiple model options, usage tracking, and integration with Weave for tracing and evaluation.

- [Introduction]({{< relref path="/guides/inference/" lang="ko" >}})
- [Available Models]({{< relref path="/guides/inference/models/" lang="ko" >}})
- [API Reference]({{< relref path="/guides/inference/api-reference/" lang="ko" >}})
- [Try in Playground](https://wandb.ai/inference)

</div>{{% /card %}}
{{< /cardpane >}}
</div>

<div class="bottom-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides/core/'" style="cursor: pointer; padding-left: 20px">
<h2>Core Components</h2>

Both W&B products share common components that enable and accelerate your AI/ML engineering work. 

- [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})
- [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})
- [Automations]({{< relref path="/guides/core/automations/" lang="ko" >}})
- [Secrets]({{< relref path="/guides/core/secrets.md" lang="ko" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/hosting'" style="cursor: pointer;padding-left:20px;">

<h2>Platform</h2>

The Weights & Biases platform can be accessed through our SaaS offering or deployed on-premise, and it provides IAM, security, monitoring, and privacy features.

- [Deployment Options]({{< relref path="/guides/hosting/hosting-options/" lang="ko" >}})
- [Identity and access management (IAM)]({{< relref path="/guides/hosting/iam/" lang="ko" >}})
- [Data Security]({{< relref path="/guides/hosting/data-security/" lang="ko" >}})
- [Privacy settings]({{< relref path="/guides/hosting/privacy-settings/" lang="ko" >}})
- [Monitoring and Usage]({{< relref path="/guides/hosting/monitoring-usage/" lang="ko" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/support/'" style="cursor: pointer;padding-left:20px;">

<h2>Support</h2>

Get help with any aspect of the Weights & Biases platform. Find answers to common questions, troubleshooting guides, and ways to contact our support team.

- [Knowledge Base Articles]({{< relref path="/support/" lang="ko" >}})
- [Community Forum](https://wandb.ai/community)
- [Discord Server](https://discord.com/invite/RgB8CPk2ce)
- [Contact Support](https://wandb.ai/site/contact/)

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

/* Make all cards uniform size in 3x2 grid */
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