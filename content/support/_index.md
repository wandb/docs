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
type: docs
cascade:
- url: support/:filename
---
import Card from '@site/src/components/Card';
import TopicCard from '@site/src/components/TopicCard';
import TopicGrid from '@site/src/components/TopicGrid';


Search for help from support articles, product documentation, and the W&B community. 

<Card className="help-banner">
  <h2>How can we help?</h2>
  <p>Browse support articles or contact us directly.</p>
</Card>

## Featured articles

Here are the most commonly asked questions across all categories.

* [What does `wandb.init` do to my training process?](./wandbinit_training_process.md)
* [Does Launch support parallelization? How can I limit the resources consumed by a job?](./launch_support_parallelization_limit_resources_consumed_job.md)
* [How do I use custom CLI commands with sweeps?](./custom_cli_commands_sweeps.md)
* [Is it possible to save metrics offline and sync them to W&B later?](./same_metric_appearing_more.md)
* [How can I configure the name of the run in my training code?](./configure_name_run_training_code.md)


If you can't find what you are looking for, browse through the [popular categories](#popular-categories) below or search through articles based on categories.


## Popular categories

Browse articles by category.

<div className="card-container">
  <Card href="index_experiments" className="card card-yellow">
    <div className="card-icon-left" style={{backgroundImage: "url('/images/support/icon-running-repeat.svg')"}} />
    <div className="card-icon-right" style={{backgroundImage: "url('/images/support/icon-forward-next.svg')"}} />
    <h2 className="card-title">Experiments</h2>
    <p className="card-content">Track, visualize, and compare machine learning experiments</p>
  </Card>

  <Card href="index_artifacts" className="card card-pink">
    <div className="card-icon-left" style={{backgroundImage: "url('/images/support/icon-versions-layers.svg')"}} />
    <div className="card-icon-right" style={{backgroundImage: "url('/images/support/icon-forward-next.svg')"}} />
    <h2 className="card-title">Artifacts</h2>
    <p className="card-content">Version and track datasets, models, and other machine learning artifacts</p>
  </Card>
</div>

<div className="card-container">

  <Card href="index_reports" className="card card-gray">
    <div className="card-icon-left" style={{backgroundImage: "url('/images/support/icon-category-multimodal.svg')"}} />
    <div className="card-icon-right" style={{backgroundImage: "url('/images/support/icon-forward-next.svg')"}} />
    <h2 className="card-title">Reports</h2>
    <p className="card-content">Create interactive, collaborative reports to share your work</p>
  </Card>
  <Card href="index_launch" className="card card-blue">
    <div className="card-icon-left" style={{backgroundImage: "url('/images/support/white-icon-category-multimodal.svg')"}} />
    <div className="card-icon-right" style={{backgroundImage: "url('/images/support/white-icon-forward-next.svg')"}} />
    <h2 className="card-title">Launch</h2>
    <p className="card-content">Manage compute resources and run machine learning jobs at scale</p>
  </Card>  
</div>

<Card className="card-banner card-banner-gray">
  <div className="card-banner-icon">
    <img src="/images/support/callout-icon.svg" alt="Callout Icon" width="32" height="32" />
  </div>
  <h2>Still can't find what you are looking for?</h2>
  <a href="mailto:support@wandb.com" className="contact-us-button">
    Contact support
  </a>
</Card>