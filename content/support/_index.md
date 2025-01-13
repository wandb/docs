---
title: Support
menu:
  support:
    identifier: support
    parent: null
  main:
    weight: 4
    parent: null
url: support
no_list: true
type: docs
cascade:
- url: support/:filename
---

{{< banner title="How can we help?" background="/images/support/support_banner.png" >}}
Search for help from support articles, product documentation,<br>
and the W&B community. 
{{< /banner >}}

## Featured articles

Here are the most commonly asked questions across all categories.

* [What does `wandb.init` do to my training process?](./wandbinit_training_process.md)
* [How do I use custom CLI commands with sweeps?](./custom_cli_commands_sweeps.md)
* [Is it possible to save metrics offline and sync them to W&B later?](./same_metric_appearing_more.md)
* [How can I configure the name of the run in my training code?](./configure_name_run_training_code.md)


If you can't find what you are looking for, browse through the [popular categories](#popular-categories) below or search through articles based on categories.


## Popular categories

Browse articles by category.

{{< cardpane >}}
  {{< card >}}
    <a href="index_experiments">
      <div className="card-icon-left" style="backgroundImage: url('/images/support/icon-running-repeat.svg')"></div>
      <div className="card-icon-right" style="backgroundImage: url('/images/support/icon-forward-next.svg')"></div>
      <h2 className="card-title">Experiments</h2>
      <p className="card-content">Track, visualize, and compare machine learning experiments</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_artifacts">
      <div className="card-icon-left" style="backgroundImage: url('/images/support/icon-versions-layers.svg')"></div>
      <div className="card-icon-right" style="backgroundImage: url('/images/support/icon-forward-next.svg')"></div>
      <h2 className="card-title">Artifacts</h2>
      <p className="card-content">Version and track datasets, models, and other machine learning artifacts</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}
{{< cardpane >}}
  {{< card >}}
    <a href="index_reports">
      <div className="card-icon-left" style="backgroundImage: url('/images/support/icon-category-multimodal.svg')"></div>
      <div className="card-icon-right" style="backgroundImage: url('/images/support/icon-forward-next.svg')"></div>
      <h2 className="card-title">Reports</h2>
      <p className="card-content">Create interactive, collaborative reports to share your work</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_sweeps">
      <div className="card-icon-left" style="backgroundImage: url('/images/support/white-icon-category-multimodal.svg')"></div>
      <div className="card-icon-right" style="backgroundImage: url('/images/support/white-icon-forward-next.svg')"></div>
      <h2 className="card-title">Sweeps</h2>
      <p className="card-content">Automate hyperparameter search</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}


{{< card >}}
  <div className="card-banner-icon">
    <img src="/images/support/callout-icon.svg" alt="Callout Icon" width="32" height="32" />
  </div>
  <h2>Still can't find what you are looking for?</h2>
  <a href="mailto:support@wandb.com" className="contact-us-button">
    Contact support
  </a>
 {{< /card >}}
