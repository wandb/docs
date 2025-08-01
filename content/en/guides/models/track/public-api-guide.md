---
description: Import data from MLFlow, export or update data that you have saved to
  W&B
menu:
  default:
    identifier: public-api-guide
    parent: experiments
title: Import and export data
weight: 8
---

Export data or import data  with W&B Public APIs.

{{% alert %}}
This feature requires python>=3.8
{{% /alert %}}

## Import data from MLFlow

W&B supports importing data from MLFlow, including experiments, runs, artifacts, metrics, and other metadata.

Install dependencies:

```shell
# note: this requires py38+
pip install wandb[importers]
```

Log in to W&B. Follow the prompts if you have not logged in before.

```shell
wandb login
```

Import all runs from an existing MLFlow server:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

By default, `importer.collect_runs()` collects all runs from the MLFlow server. If you prefer to upload a special subset, you can construct your own runs iterable and pass it to the importer.

```py
import mlflow
from wandb.apis.importers.mlflow import MlflowRun

client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)

runs: Iterable[MlflowRun] = []
for run in mlflow_client.search_runs(...):
    runs.append(MlflowRun(run, client))

importer.import_runs(runs)
```

{{% alert %}}
You might need to [configure the Databricks CLI first](https://docs.databricks.com/dev-tools/cli/index.html) if you import from Databricks MLFlow.

Set `mlflow-tracking-uri="databricks"` in the previous step.
{{% /alert %}}

To skip importing artifacts, you can pass `artifacts=False`:

```py
importer.import_runs(runs, artifacts=False)
```

To import to a specific W&B entity and project, you can pass a `Namespace`:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

<!-- Per DOCS-1043, hiding this information until it gets fixed 

## Import data from another W&B instance

{{% alert %}}
This feature is in beta, and only supports importing from the W&B public cloud.
{{% /alert %}}

Install dependencies:

```sh
# note: this requires py38+
pip install wandb[importers]
```

Log in to the source W&B server. Follow the prompts if you have not logged in before.

```sh
wandb login
```

Import all runs and artifacts from a source W&B instance to a destination W&B instance. Runs and artifacts are imported to their respective namespaces in the destination instance.

```py
from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace

importer = WandbImporter(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-api-key-here",
    dst_base_url="https://example-target.wandb.io",
    dst_api_key="target-environment-api-key-here",
)

# Imports all runs, artifacts, reports
# from "entity/project" in src to "entity/project" in dst
importer.import_all(namespaces=[
    Namespace(entity, project),
    # ... add more namespaces here
])
```

If you prefer to change the destination namespace, you can specify `remapping: dict[Namespace, Namespace]`

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    remapping={
        Namespace(entity, project): Namespace(new_entity, new_project),
    }
)
```

By default, imports are incremental. Subsequent imports try to validate the previous work and write to `.jsonl` files tracking success/failure. If an import succeeded, future validation is skipped. If an import failed, it is retried. To turn this off, set `incremental=False`.

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    incremental=False,
)
```

### Known issues and limitations

- If the destination namespace does not exist, W&B creates one automatically.
- If a run or artifact has the same ID in the destination namespace, W&B treats it as an incremental import. The destination run/artifact is validated and retried if it failed in a previous import.
- No data is ever deleted from the source system.

1. Sometimes when bulk importing (especially large artifacts), you can run into S3 rate limits. If you see `botocore.exceptions.ClientError: An error occurred (SlowDown) when calling the PutObject operation`, you can try spacing out imports by moving just a few namespaces at a time.
2. Imported run tables appear to be blank in the workspace, but if you nav to the Artifacts tab and click the equivalent run table artifact you should see the table as expected.
3. System metrics and custom charts (not explicitly logged with `run.log`) are not imported

-->

## Export Data

Use the Public API to export or update data that you have saved to W&B. Before using this API, log data from your script. Check the [Quickstart]({{< relref "/guides/quickstart.md" >}}) for more details.

**Use Cases for the Public API**

- **Export Data**: Pull down a dataframe for custom analysis in a Jupyter Notebook. Once you have explored the data, you can sync your findings by creating a new analysis run and logging results, for example: `wandb.init(job_type="analysis")`
- **Update Existing Runs**: You can update the data logged in association with a W&B run. For example, you might want to update the config of a set of runs to include additional information, like the architecture or a hyperparameter that wasn't originally logged.

See the [Generated Reference Docs]({{< relref "/ref/python/public-api/" >}}) for details on available functions.

### Create an API key

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to the [W&B authorization page](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.


### Find the run path

To use the Public API, you'll often need the run path which is `<entity>/<project>/<run_id>`. In the app UI, open a run page and click the [Overview tab ]({{< relref "/guides/models/track/runs/#overview-tab" >}})to get the run path.


### Export Run Data

Download data from a finished or active run. Common usage includes downloading a dataframe for custom analysis in a Jupyter notebook, or using custom logic in an automated environment.

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

The most commonly used attributes of a run object are:

| Attribute       | Meaning                                                                                                                                                                                                                                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | A dictionary of the run's configuration information, such as the hyperparameters for a training run or the preprocessing methods for a run that creates a dataset Artifact. Think of these as the run's inputs.                                                                                                    |
| `run.history()` | A list of dictionaries meant to store values that change while the model is training such as loss. The command `run.log()` appends to this object.                                                                                                                                                                 |
| `run.summary`   | A dictionary of information that summarizes the run's results. This can be scalars like accuracy and loss, or large files. By default, `run.log()` sets the summary to the final value of a logged time series. The contents of the summary can also be set directly. Think of the summary as the run's outputs. |

You can also modify or update the data of past runs. By default a single instance of an api object will cache all network requests. If your use case requires real time information in a running script, call `api.flush()` to get updated values.

### Understanding different run attributes

The following code snippet shows how to create a run, log some data, and then access the run's attributes:

```python
import wandb
import random

with wandb.init(project="public-api-example") as run:
    n_epochs = 5
    config = {"n_epochs": n_epochs}
    run.config.update(config)
    for n in range(run.config.get("n_epochs")):
        run.log(
            {"val": random.randint(0, 1000), "loss": (random.randint(0, 1000) / 1000.00)}
        )
```

The following sections describe the different outputs for the above run object attributes

##### `run.config`

```python
{"n_epochs": 5}
```
    
#### `run.summary`

```python
{
    "_runtime": 4,
    "_step": 4,
    "_timestamp": 1644345412,
    "_wandb": {"runtime": 3},
    "loss": 0.041,
    "val": 525,
}
```

### Sampling

The default history method samples the metrics to a fixed number of samples (the default is 500, you can change this with the `samples` __ argument). If you want to export all of the data on a large run, you can use the `run.scan_history()` method. For more details see the [API Reference]({{< relref "/ref/python/public-api" >}}).

### Querying Multiple Runs

{{< tabpane text=true >}}
    {{% tab header="DataFrame and CSVs" %}}
This example script finds a project and outputs a CSV of runs with name, configs and summary stats. Replace `<entity>` and `<project>` with your W&B entity and the name of your project, respectively.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")

run.finish()
```
    {{% /tab %}}
    {{% tab header="MongoDB Style" %}}
The W&B API also provides a way for you to query across runs in a project with api.runs(). The most common use case is exporting runs data for custom analysis. The query interface is the same as the one [MongoDB uses](https://docs.mongodb.com/manual/reference/operator/query).

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```    
    {{% /tab %}}
{{< /tabpane >}}


Calling `api.runs` returns a `Runs` object that is iterable and acts like a list. By default the object loads 50 runs at a time in sequence as required, but you can change the number loaded per page with the `per_page` keyword argument.

`api.runs` also accepts an `order` keyword argument. The default order is `-created_at`. To order results ascending, specify `+created_at`. You can also sort by config or summary values. For example, `summary.val_acc` or `config.experiment_name`.

### Error Handling

If errors occur while talking to W&B servers a `wandb.CommError` will be raised. The original exception can be introspected via the `exc` attribute.

### Get the latest git commit through the API

In the UI, click on a run and then click the Overview tab on the run page to see the latest git commit. It's also in the file `wandb-metadata.json` . Using the public API, you can get the git hash with `run.commit`.

### Get a run's name and ID during a run

After calling `wandb.init()` you can access the random run ID or the human readable run name from your script like this:

- Unique run ID (8 character hash): `run.id`
- Random run name (human readable): `run.name`

If you're thinking about ways to set useful identifiers for your runs, here's what we recommend:

- **Run ID**: leave it as the generated hash. This needs to be unique across runs in your project.
- **Run name**: This should be something short, readable, and preferably unique so that you can tell the difference between different lines on your charts.
- **Run notes**: This is a great place to put a quick description of what you're doing in your run. You can set this with `wandb.init(notes="your notes here")`
- **Run tags**: Track things dynamically in run tags, and use filters in the UI to filter your table down to just the runs you care about. You can set tags from your script and then edit them in the UI, both in the runs table and the overview tab of the run page. See the detailed instructions [here]({{< relref "/guides/models/track/runs/tags.md" >}}).

## Public API Examples

### Export data to visualize in matplotlib or seaborn

Check out our [API examples]({{< relref "/ref/python/public-api/" >}}) for some common export patterns. You can also click the download button on a custom plot or on the expanded runs table to download a CSV from your browser.

### Read metrics from a run

This example outputs timestamp and accuracy saved with `run.log({"accuracy": acc})` for a run saved to `"<entity>/<project>/<run_id>"`.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Filter runs

You can filters by using the MongoDB Query Language.

#### Date

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### Read specific metrics from a run

To pull specific metrics from a run, use the `keys` argument. The default number of samples when using `run.history()` is 500. Logged steps that do not include a specific metric will appear in the output dataframe as `NaN`. The `keys` argument will cause the API to sample steps that include the listed metric keys more frequently.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Compare two runs

This will output the config parameters that are different between `run1` and `run2`.

```python
import pandas as pd
import wandb

api = wandb.Api()

# replace with your <entity>, <project>, and <run_id>
run1 = api.run("<entity>/<project>/<run_id>")
run2 = api.run("<entity>/<project>/<run_id>")


df = pd.DataFrame([run1.config, run2.config]).transpose()

df.columns = [run1.name, run2.name]
print(df[df[run1.name] != df[run2.name]])
```

Outputs:

```
              c_10_sgd_0.025_0.01_long_switch base_adam_4_conv_2fc
batch_size                                 32                   16
n_conv_layers                               5                    4
optimizer                             rmsprop                 adam
```

### Update metrics for a run, after the run has finished

This example sets the accuracy of a previous run to `0.9`. It also modifies the accuracy histogram of a previous run to be the histogram of `numpy_array`.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### Rename a metric in a completed run

This example renames a summary column in your tables.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
Renaming a column only applies to tables. Charts will still refer to metrics by their original names.
{{% /alert %}}



### Update config for an existing run

This examples updates one of your configuration settings.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### Export system resource consumptions to a CSV file

The snippet below would find the system resource consumptions and then, save them to a CSV.

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### Get unsampled metric data

When you pull data from history, by default it's sampled to 500 points. Get all the logged data points using `run.scan_history()`. Here's an example downloading all the `loss` data points logged in history.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### Get paginated data from history

If metrics are being fetched slowly on our backend or API requests are timing out, you can try lowering the page size in `scan_history` so that individual requests don't time out. The default page size is 500, so you can experiment with different sizes to see what works best:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### Export metrics from all runs in a project to a CSV file

This script pulls down the runs in a project and produces a dataframe and a CSV of runs including their names, configs, and summary stats. Replace `<entity>` and `<project>` with your W&B entity and the name of your project, respectively.

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### Get the starting time for a run

This code snippet retrieves the time at which the run was created.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### Upload files to a finished run

The code snippet below uploads a selected file to a finished run.

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### Download a file from a run

This finds the file "model-best.h5" associated with run ID uxte44z7 in the cifar project and saves it locally.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### Download all files from a run

This finds all files associated with a run and saves them locally.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### Get runs from a specific sweep

This snippet downloads all the runs associated with a particular sweep.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### Get the best run from a sweep

The following snippet gets the best run from a given sweep.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

The `best_run` is the run with the best metric as defined by the `metric` parameter in the sweep config.

### Download the best model file from a sweep

This snippet downloads the model file with the highest validation accuracy from a sweep with runs that saved model files to `model.h5`.

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
val_acc = runs[0].summary.get("val_acc", 0)
print(f"Best run {runs[0].name} with {val_acc}% val accuracy")

runs[0].file("model.h5").download(replace=True)
print("Best model saved to model-best.h5")
```

### Delete all files with a given extension from a run

This snippet deletes files with a given extension from a run.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

extension = ".png"
files = run.files()
for file in files:
    if file.name.endswith(extension):
        file.delete()
```

### Download system metrics data

This snippet produces a dataframe with all the system resource consumption metrics for a run and then saves it to a CSV.

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### Update summary metrics

You can pass a dictionary to update summary metrics.

```python
summary.update({"key": val})
```

### Get the command that ran the run

Each run captures the command that launched it on the run overview page. To pull this command down from the API, you can run:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```