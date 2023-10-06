---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face Transformers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb"></CTAButtons>

The [Hugging Face Transformers](https://huggingface.co/transformers/) library makes state-of-the-art NLP models like BERT and training techniques like mixed precision and gradient checkpointing easy to use. The [W&B integration](https://huggingface.co/transformers/main\_classes/callback.html#transformers.integrations.WandbCallback) adds rich, flexible experiment tracking and model versioning to interactive centralized dashboards without compromising that ease of use.

## 🤗 Next-level logging in few lines

```python
os.eviron["WANDB_PROJECT"] = "<my-amazing-project>" # log to your project 
os.eviron["WANDB_LOG_MODEL"] = "all" # log your models

from transformers import TrainingArguments, Trainer

args = TrainingArguments(... , report_to="wandb")
trainer = Trainer(... , args=args)
```
![Explore your experiment results in the W&B interactive dashboard](@site/static/images/integrations/huggingface_gif.gif)

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

### 4) Turn on model Checkpoints and Versioning


<summary>How to Fine-Tune Hugging Face Transformers on a Custom Dataset</summary>

We fine tune a DistilBERT transformer for sentiment analysis (binary classification) on a custom IMDB dataset.

Read the full report [here](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc).
</details>

## Advanced features

### Turn on model versioning

Using [Weights & Biases' Artifacts](https://docs.wandb.ai/artifacts), you can store up to 100GB of models and datasets. Logging your Hugging Face model to W&B Artifacts can be done by setting a W&B environment variable called `WANDB_LOG_MODEL` to one of `end`, `checkpoint` or `"false"`. If set to "end", the model will be uploaded at the end of training. If set to "checkpoint", the checkpoint will be uploaded every `args.save_steps` from the [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments). Use along with `load_best_model_at_end` to upload the best model.


<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash

WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
</Tabs>


:::info
By default, your model will be saved to W&B Artifacts as `model-{run_id}` when `WANDB_LOG_MODEL` is set to `end` or `checkpoint-{run_id}` when `WANDB_LOG_MODEL` is set to `checkpoint`.
However, If you pass a [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name) in your `TrainingArguments`, the model will be saved as `model-{run_name}` or `checkpoint-{run_name}`.
:::


Any `Trainer` you initialize from now on will upload models to your W&B project.

The model checkpoints you log will be viewable through the [W&B Artifacts](https://docs.wandb.ai/guides/artifacts) UI, and include the full model lineage (see an example model checkpoint in the UI [here](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..). 

To bookmark your best model checkpoints and centralize them across your team, you can link them to the [W&B Model Registry](https://docs.wandb.ai/guides/models). Here you can organize your best models by task, manage model lifecycle, facilitate easy tracking and auditing throughout the ML lifecyle, and [automate](https://docs.wandb.ai/guides/models/automation) downstream actions with webhooks or jobs.
 

#### (Notebook only) Finish your W&B Run

If your training is encapsulated in a Python script, the W&B run will end when your script finishes.

If you are using a Jupyter or Google Colab notebook, you'll need to tell us when you're done with training by calling `wandb.finish()`.

```python
trainer.train()  # start training and logging to W&B

# post-training analysis, testing, other logged code

wandb.finish()
```

### 5) Visualize your results

Once you have logged your training results you can explore your results dynamically in the [W&B Dashboard](../track/app.md). It's easy to compare across dozens of runs at once, zoom in on interesting findings, and coax insights out of complex data with flexible, interactive visualizations.

## Advanced features and FAQs

### How do I save the best model?
Want to centralize all your best model versions across your team to organize them by ML task, stage them for production, bookmark them for further evaluation, or kick off downstream Model CI/CD processes?
Check out the [Model Registry](../model_registry/intro.md)

If `load_best_model_at_end=True` is passed to `Trainer`, then W&B will save the best performing model to Artifacts.

### Loading a saved model

If you saved your model to W&B Artifacts with `WANDB_LOG_MODEL`, you can download your model weights for additional training or to run inference. You just load them back into the same Hugging Face architecture that you used before.

```python
# Create a new run
with wandb.init(project="amazon_sentiment_analysis") as run:

  # Connect an Artifact to the run
  my_model_name = "model-bert-base-high-lr:latest"
  my_model_artifact = run.use_artifact(my_model_name)

  # Download model weights to a folder and return the path
  model_dir = my_model_artifact.download()

  # Load your Hugging Face model from that folder
  #  using the same model class
  model = AutoModelForSequenceClassification.from_pretrained(
      model_dir, num_labels=num_labels)

  # Do additional training, or run inference
```

### Resume training from a checkpoint 
If you had set `WANDB_LOG_MODEL='checkpoint'` you can also resume training by you can using the `model_dir` as the `model_name_or_path` argument in your `TrainingArguments` and pass `resume_from_checkpoint=True` to `Trainer`.

```python
last_run_id = "xxxxxxxx"  # fetch the run_id from your wandb workspace

# resume the wandb run from the run_id
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",) as run:
    
  # Connect an Artifact to the run
  my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
  my_checkpoint_artifact = run.use_artifact(my_model_name)
  
  # Download checkpoint to a folder and return the path
  checkpoint_dir = my_checkpoint_artifact.download()
  
  # reinitialize your model and trainer
  model = AutoModelForSequenceClassification.from_pretrained(
      <model_name>, num_labels=num_labels)
  # your awesome training arguments here.
  training_args = TrainingArguments(...) 
  
  trainer = Trainer(
      model=model,
      args=training_args,
      ...)
  
  # make sure use the checkpoint dir to resume training from the checkpoint
  trainer.train(resume_from_checkpoint=checkpoint_dir) 
```


### Additional W&B settings

Further configuration of what is logged with `Trainer` is possible by setting environment variables. A full list of W&B environment variables [can be found here](https://docs.wandb.ai/library/environment-variables).

| Environment Variable | Usage                                                                                                                                                                                                                                                                                                    |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | Give your project a name (`huggingface` by default)                                                                                                                                                                                                                                                      |
| `WANDB_LOG_MODEL`    | Log the model as artifact at the end of training (`false` by default)                                                                                                                                                                                                                                    |
| `WANDB_WATCH`        | <p>Set whether you'd like to log your models gradients, parameters or neither</p><ul><li><code>false</code> (default): No gradient or parameter logging </li><li><code>gradients</code>: Log histograms of the gradients </li><li><code>all</code>: Log histograms of gradients and parameters</li></ul> |
| `WANDB_DISABLED`     | Set to `true` to disable logging entirely (`false` by default)                                                                                                                                                                                                                                           |
| `WANDB_SILENT`       | Set to `true` to silence the output printed by wandb (`false` by default)                                                                                                                                                                                                                                |

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
The following section shows how to customize the `WandbCallback` to run model predictions and log evaluation samples during training.

### View evaluation samples during training
For instance here's how you could modify the callback to log predictions to a Weights & Biases table after each evaluation

In the following example, we log the predictions of a language model every `eval_steps`.
We can use the `on_evaluate` method of the callback to log the predictions to a Weights & Biases table.
Here, we use the `decode_predictions` function to decode the predictions and labels from the model output using the tokenizer.
Then, we create a pandas DataFrame from the predictions and labels and add an `epoch` column to the DataFrame.
Finally, we create a `wandb.Table` from the DataFrame and log it to wandb.
Additionally, we can control the frequency of logging by logging the predictions every `freq` epochs.

**Note**: Unlike the regular `WandbCallback` this custom callback needs to be added to the trainer **after** the `Trainer` is instantiated and not during initialization of the `Trainer`.
This is because the `Trainer` instance is passed to the callback during initialization.

```python
from transformers.integrations import WandbCallback
import pandas as pd

def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    prediction_text = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1))
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq


    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs
        if state.epoch % self.freq == 0:
          # generate predictions
          predictions = self.trainer.predict(self.sample_dataset)
          # decode predictions and labels
          predictions = decode_predictions(self.tokenizer, predictions)
          # add predictions to a wandb.Table
          predictions_df = pd.DataFrame(predictions)
          predictions_df["epoch"] = state.epoch
          records_table = self._wandb.Table(dataframe=predictions_df)
          # log the table to wandb
          self._wandb.log({"sample_predictions": records_table})

# First, instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],

)

# Instantiate the WandbPredictionProgressCallback
progress_callback = WandbPredictionProgressCallback(trainer, tokenizer, lm_dataset["validation"], 10, 2)
# Add the callback to the trainer
trainer.add_callback(progress_callback)

```

For a more detailed example please refer to this [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb)


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

## Issues, questions, feature requests

For any issues, questions, or feature requests for the Hugging Face W&B integration, feel free to post in [this thread on the Hugging Face forums](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498) or open an issue on the Hugging Face [Transformers GitHub repo](https://github.com/huggingface/transformers).
