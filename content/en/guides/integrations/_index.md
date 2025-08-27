---
menu:
  default:
    identifier: integrations
title: Integrations
weight: 9
url: guides/integrations
cascade:
- url: guides/integrations/:filename
no_list: true
---

<!-- W&B integrations make it fast and easy to set up experiment tracking and data versioning inside existing projects. Check out integrations for ML frameworks such as [PyTorch]({{< relref "pytorch.md" >}}), ML libraries such as [Hugging Face]({{< relref "huggingface.md" >}}), or cloud services such as [Amazon SageMaker]({{< relref "sagemaker.md" >}}).


<iframe width="668" height="376" src="https://www.youtube.com/embed/hmewPDNUNJs?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk" title="Log Your First Run With W&amp;B" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Related resources

* [Examples](https://github.com/wandb/examples): Try the code with notebook and script examples for each integration.
* [Video Tutorials](https://www.youtube.com/playlist?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk): Learn to use W&B with YouTube video tutorials -->

Use W&B with popular ML frameworks and libraries.

## Popular frameworks

<div class="top-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important; margin-bottom: -10px !important">
<img src="/img/pytorch-logo.png" alt="PyTorch logo" width="50" height="50"/>
</div>
<h2>PyTorch</h2>

Explore W&B integration for various PyTorch libraries:

- [PyTorch]({{< relref "/guides/integrations/pytorch.md" >}})
- [PyTorch Lightning]({{< relref "/guides/integrations/lightning.md" >}})
- [PyTorch Ignite]({{< relref "/guides/integrations/ignite.md" >}})
- [PyTorch Geometric]({{< relref "/guides/integrations/pytorch-geometric.md" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important; margin-bottom: -10px !important">
<img src="/img/hf-logo.png" alt="Hugging Face logo" width="60" height="60"/>
</div>
<h2>HuggingFace</h2>

Explore W&B integration for various Hugging Face libraries:

- [Hugging Face Accelerate]({{< relref "/guides/integrations/accelerate" >}})
- [Hugging Face Diffusers]({{< relref "/guides/integrations/diffusers" >}})
- [Hugging Face Transformers]({{< relref "/guides/integrations/huggingface" >}})

</div>{{% /card %}}
{{< /cardpane >}}


<div class="top-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important; margin-bottom: -10px !important">
<!-- <img src="/img/openai-logo.png" alt="OpenAI logo" width="50" height="50"/> -->
</div>
<h2>OpenAI</h2>

Explore W&B integration for various OpenAI libraries:

- [OpenAI API]({{< relref "/guides/integrations/openai-api.md" >}})
- [OpenAI Gym]({{< relref "/guides/integrations/openai-gym.md" >}})
- [OpenAI Fine-Tuning]({{< relref "/guides/integrations/openai-fine-tuning.md" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important; margin-bottom: -10px !important">
<!-- <img src="/img/hf-logo.png" alt="Keras logo" width="60" height="60"/> -->
</div>
<h2>XGBoost</h2>

Explore W&B integration for [XGBoost]({{< relref "/guides/integrations/xgboost.md" >}}) library.

</div>{{% /card %}}
{{< /cardpane >}}


{{< card >}}
  <div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
    {{< img src="/images/support/callout-icon.svg" alt="Callout Icon" width="32" height="32" >}}
  </div>
  <h2>Add W&B to your own custom library</h2>

Learn how to integrate W&B into your own Python library to get powerful Experiment Tracking, GPU and System Monitoring, Model Checkpointing, and more for your own library.
 {{< /card >}}
 
<div style="padding-top:50px;">&nbsp;</div>

<!-- Supported Libraries


- Simple Transformers
- Keras
- RayTune
- XGBoost
- YOLOv5 -->

<!-- Option 2  -->

## Popular frameworks

{{< cardpane >}}
  {{< card >}}
    <a href="/support/experiments">
      <h2 className="card-title">PyTorch</h2>
    </a>
    <p className="card-content">Track, visualize, and compare machine learning experiments</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="/support/artifacts">
      <h2 className="card-title">Hugging Face</h2>
    </a>
    <p className="card-content">Version and track datasets, models, and other machine learning artifacts</p>
  {{< /card >}}
{{< /cardpane >}}
{{< cardpane >}}
  {{< card >}}
    <a href="/support/reports">
      <h2 className="card-title">Open AI</h2>
    </a>
    <p className="card-content">Create interactive, collaborative reports to share your work</p>
  {{< /card >}}
  {{< card >}}
    <a href="/support/sweeps">
      <h2 className="card-title">TensorFlow</h2>
    </a>
    <p className="card-content">Automate hyperparameter search</p>
  {{< /card >}}
{{< /cardpane >}}


{{< card >}}
  <div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
    {{< img src="/images/support/callout-icon.svg" alt="Callout Icon" width="32" height="32" >}}
  </div>
  <h2>Add W&B to your own custom library</h2>

Learn how to integrate W&B into your own Python library to get powerful Experiment Tracking, GPU and System Monitoring, Model Checkpointing, and more for your own library.
 {{< /card >}}



<div style="padding-top:50px;">&nbsp;</div>
