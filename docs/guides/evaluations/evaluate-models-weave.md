---
title: Evaluate models with Weave
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

## What is Weave? 

W&B Weave helps developers who are building and iterating on their AI applications to create apples-to-apples evaluations that score the behavior of any aspect of their app, and examine and debug failures by easily inspecting inputs and outputs.

## How do I get started with Weave? 

First, create a W&B account at https://wandb.ai and copy your API key from https://wandb.ai/authorize

Then, you can follow along in the below Colab notebook that demonstrates Weave evaluating an LLM (in this case, OpenAI).

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/weave/blob/master/docs/intro_notebook.ipynb'/>

After running through the steps you will be able to browse your dashboard in Weave and see some of the tracing data that is generated when executing your code that includes calls to your LLM, and see breakdowns of execution time, cost, etc. 

![](https://weave-docs.wandb.ai/assets/images/weave-hero-188bbbbfcac1809f2529c62110d1553a.png)

## How do I use Weave to evaluate models in production? 

This [tutorial on how to build an evaluation pipeline with Weave](https://weave-docs.wandb.ai/tutorial-eval/) can help, which demonstrates how multiple versions of an application that uses a model is evolving. In the tutorial you'll see how the `weave.Evaluation` function assess a Models performance on a set of examples using a list of specified scoring functions or `weave.scorer.Scorer` classes, producing dashboards with advanced breakdowns of the model's performance.

![](https://weave-docs.wandb.ai/assets/images/evals-hero-9bb44591b72ac8637e7e14bc73db1ba8.png)