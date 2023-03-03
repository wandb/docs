import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hugging Face Transformers

The [Hugging Face Transformers](https://huggingface.co/transformers/) library makes state-of-the-art NLP models like BERT and training techniques like mixed precision and gradient checkpointing easy to use. The [W&B integration](https://huggingface.co/transformers/main\_classes/callback.html#transformers.integrations.WandbCallback) adds rich, flexible experiment tracking and model versioning to interactive centralized dashboards without compromising that ease of use.

## 🤗 Next-level logging in 2 lines

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(... , report_to="wandb")
trainer = Trainer(... , args=args)
```
![Explore your experiment results in the W&B interactive dashboard](@site/static/images/integrations/huggingface_gif.gif)

## This guide covers

* how to [**get started using W&B with Hugging Face Transformers**](huggingface.md#getting-started-track-and-save-your-models) to track your NLP experiments and
* how to use [**advanced features of the W&B Hugging Face integration**](../track/intro.md) to get the most out of experiment tracking.

:::info
If you'd rather dive straight into working code, check out this [Google Colab](https://wandb.me/hf).
:::

## Getting started: track experiments

### 1) Sign Up, install the `wandb` library and log in

a) [**Sign up**](https://wandb.ai/site) for a free account

b) Pip install the `wandb` library

c) To login in your training script, you'll need to be signed in to you account at www.wandb.ai, then **you will find your API key on the** [**Authorize page**](https://wandb.ai/authorize)**.**

If you are using Weights and Biases for the first time you might want to check out our [**quickstart**](../../quickstart.md)

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

### 2) Name the project

A [Project](../app/pages/project-page.md) is where all of the charts, data, and models logged from related runs are stored. Naming your project helps you organize your work and keep all the information about a single project in one place.

To add a run to a project simply set the `WANDB_PROJECT` environment variable to the name of your project. The `WandbCallback` will pick up this project name environment variable and use it when setting up your run.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
</Tabs>


:::info
Make sure you set the project name _before_ you initialize the `Trainer`.
:::

If a project name is not specified the project name defaults to "huggingface".

### 3) Log your training runs to W&B

This is **the most important step:** when defining your `Trainer` training arguments, either inside your code or from the command line, set `report_to` to `"wandb"` in order enable logging with Weights & Biases.

You can also give a name to the training run using the `run_name` argument.

:::info
Using TensorFlow? Just swap the PyTorch `Trainer` for the TensorFlow `TFTrainer`.
:::

That's it! Now your models will log losses, evaluation metrics, model topology, and gradients to Weights & Biases while they train.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
python run_glue.py \     # run your Python script
  --report_to wandb \    # enable logging to W&B
  --run_name bert-base-high-lr \   # name of the W&B run (optional)
  # other command line arguments here
```

  </TabItem>
  <TabItem value="notebook">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # other args and kwargs here
    report_to="wandb",  # enable logging to W&B
    run_name="bert-base-high-lr"  # name of the W&B run (optional)
)

trainer = Trainer(
    # other args and kwargs here
    args=args,  # your training args
)

trainer.train()  # start training and logging to W&B
```

  </TabItem>
</Tabs>

#### (Notebook only) Finish your W&B Run

If your training is encapsulated in a Python script, the W&B run will end when your script finishes.

If you are using a Jupyter or Google Colab notebook, you'll need to tell us when you're done with training by calling `wandb.finish()`.

```python
trainer.train()  # start training and logging to W&B

# post-training analysis, testing, other logged code

wandb.finish()
```

### 4) Visualize your results

Once you have logged your training results you can explore your results dynamically in the [W&B Dashboard](../track/app.md). It's easy to compare across dozens of runs at once, zoom in on interesting findings, and coax insights out of complex data with flexible, interactive visualizations.

## Highlighted Articles

Below are 6 Transformers and W&B related articles you might enjoy

<details>

<summary>Hyperparameter Optimization for Hugging Face Transformers</summary>

* Three strategies for hyperparameter optimization for Hugging Face Transformers are compared - Grid Search, Bayesian Optimization, and Population Based Training.
* We use a standard uncased BERT model from Hugging Face transformers, and we want to fine-tune on the RTE dataset from the SuperGLUE benchmark
* Results show that Population Based Training is the most effective approach to hyperparameter optimization of our Hugging Face transformer model.

Read the full report [here](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI).
</details>

<details>

<summary>Hugging Tweets: Train a Model to Generate Tweets</summary>

* In the article, the author demonstrates how to fine-tune a pre-trained GPT2 HuggingFace Transformer model on anyone's Tweets in five minutes.
* The model uses the following pipeline: Downloading Tweets, Optimizing the Dataset, Initial Experiments, Comparing Losses Between Users, Fine-Tuning the Model.

Read the full report [here](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI).
</details>

<details>

<summary>Sentence Classification With Hugging Face BERT and WB</summary>

* In this article, we'll build a sentence classifier leveraging the power of recent breakthroughs in Natural Language Processing, focusing on an application of transfer learning to NLP.
* We'll be using The Corpus of Linguistic Acceptability (CoLA) dataset for single sentence classification, which is a set of sentences labeled as grammatically correct or incorrect that was first published in May 2018.
* We'll use Google's BERT to create high performance models with minimal effort on a range of NLP tasks.

Read the full report [here](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA).
</details>

<details>

<summary>A Step by Step Guide to Tracking Hugging Face Model Performance</summary>

* We use Weights & Biases and Hugging Face transformers to train DistilBERT, a Transformer that's 40% smaller than BERT but retains 97% of BERT's accuracy, on the GLUE benchmark
* The GLUE benchmark is a collection of nine datasets and tasks for training NLP models

Read the full report [here](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU).
</details>

<details>

<summary>Early Stopping in HuggingFace - Examples</summary>

* Fine-tuning a Hugging Face Transformer using Early Stopping regularization can be done natively in PyTorch or TensorFlow.
* Using the EarlyStopping callback in TensorFlow is straightforward with the `tf.keras.callbacks.EarlyStopping`callback.
* In PyTorch, there is not an off-the-shelf early stopping method, but there is a working early stopping hook available on GitHub Gist.

Read the full report [here](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM).
</details>

<details>

<summary>How to Fine-Tune Hugging Face Transformers on a Custom Dataset</summary>

We fine tune a DistilBERT transformer for sentiment analysis (binary classification) on a custom IMDB dataset.

Read the full report [here](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc).
</details>

## Advanced features

### Turn on model versioning

Using [Weights & Biases' Artifacts](https://docs.wandb.ai/artifacts), you can store up to 100GB of models and datasets. Logging your Hugging Face model to W&B Artifacts can be done by setting a W&B environment variable called `WANDB_LOG_MODEL` to `true`.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_LOG_MODEL=true
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_LOG_MODEL=true
```

  </TabItem>
</Tabs>


:::info
Your model will be saved to W&B Artifacts as `run-{run_name}`.
:::

Any `Trainer` you initialize from now on will upload models to your W&B project. Your model file will be viewable through the W&B Artifacts UI. See the [Weights & Biases' Artifacts guide](https://docs.wandb.ai/artifacts) for more about how to use Artifacts for model and dataset versioning.

#### How do I save the best model?

If `load_best_model_at_end=True` is passed to `Trainer`, then W&B will save the best performing model checkpoint to Artifacts instead of the final checkpoint.

### Loading a saved model

If you saved your model to W&B Artifacts with `WANDB_LOG_MODEL`, you can download your model weights for additional training or to run inference. You just load them back into the same Hugging Face architecture that you used before.

```python
# Create a new run
with wandb.init(project="amazon_sentiment_analysis") as run:

  # Connect an Artifact to the run
  my_model_name = "run-bert-base-high-lr:latest"
  my_model_artifact = run.use_artifact(my_model_name)

  # Download model weights to a folder and return the path
  model_dir = my_model_artifact.download()

  # Load your Hugging Face model from that folder
  #  using the same model class
  model = AutoModelForSequenceClassification.from_pretrained(
      model_dir, num_labels=num_labels)

  # Do additional training, or run inference
```

### Additional W&B settings

Further configuration of what is logged with `Trainer` is possible by setting environment variables. A full list of W&B environment variables [can be found here](https://docs.wandb.ai/library/environment-variables).

| Environment Variable | Usage                                                                                                                                                                                                                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `WANDB_PROJECT`      | Give your project a name                                                                                                                                                                                                                                                                               |
| `WANDB_LOG_MODEL`    | Log the model as artifact at the end of training (`false` by default)                                                                                                                                                                                                                                  |
| `WANDB_WATCH`        | <p>Set whether you'd like to log your models gradients, parameters or neither</p><ul><li><code>gradients</code>: Log histograms of the gradients (default)</li><li><code>all</code>: Log histograms of gradients and parameters</li><li><code>false</code>: No gradient or parameter logging</li></ul> |
| `WANDB_DISABLED`     | Set to `true` to disable logging entirely (`false` by default)                                                                                                                                                                                                                                         |
| `WANDB_SILENT`       | Set to `true` to silence the output printed by wandb (`false` by default)                                                                                                                                                                                                                              |

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

  </TabItem>
</Tabs>

### Customize `wandb.init`

The `WandbCallback` that `Trainer` uses will call `wandb.init` under the hood when `Trainer` is initialized. You can alternatively set up your runs manually by calling `wandb.init` before the`Trainer` is initialized. This gives you full control over your W&B run configuration.

An example of what you might want to pass to `init` is below. For more details on how to use `wandb.init`, [check out the reference documentation](../../ref/python/init.md).

```python
wandb.init(project="amazon_sentiment_analysis", 
           name="bert-base-high-lr",
           tags=["baseline", "high-lr"],
           group="bert")
```

### Custom logging

Logging to Weights & Biases via the [Transformers `Trainer` ](https://huggingface.co/transformers/main\_classes/trainer.html)is taken care of by the `WandbCallback` ([reference documentation](https://huggingface.co/transformers/main\_classes/callback.html#transformers.integrations.WandbCallback)) in the Transformers library. If you need to customize your Hugging Face logging you can modify this callback.

## Issues, questions, feature requests

For any issues, questions, or feature requests for the Hugging Face W&B integration, feel free to post in [this thread on the Hugging Face forums](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498) or open an issue on the Hugging Face [Transformers GitHub repo](https://github.com/huggingface/transformers).
