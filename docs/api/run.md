---
title: Single Run API
sidebar_label: Single Run API
---

## Overview

The run API lets you export and update data from previous runs.

## Examples

### Reading
This example outputs timestamp and accuracy saved with `wandb.log({"accuracy": acc})` for a run saved to `<entity>/<project>/<run_id>`.
 
```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
   for k in run.history():
       print(k["_timestamp"], k["accuracy"]) 
```

### Updating
This example sets the accuracy of a previous run to 0.9.
It also modifies the accuracy histogram of a previous run to be the histogram of numpy_arry

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```


## Api Methods

| Method     | Params                                          | Description                                                                                                               |
| ---------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| init       | _overrides={"username": None, "project": None}_ | Accepts optional setting overrides. If you specify username and project here you don't need to include them in the paths. |
| run        | _path=""_                                       | Returns a Run object given a path. If can be run_id if a global username and project is set.                              |
| runs       | _path="", filters={}_                           | Returns a Runs object given a path to a project and optional filters.                                                     |
| create_run | _project=None, username=None, run_id=None_      | Returns a new run object after creating it on the server.                                                                 |

## Run Attributes

| Attribute      | Description                                       |
| -------------- | ------------------------------------------------- |
| tags           | a list of tags associated with the run            |
| url            | the url of this run                               |
| name           | the unique identifier of the run                  |
| state          | one of: _running, finished, crashed, aborted_     |
| config         | a dict of hyperparameters associated with the run |
| created_at     | when the run was started                          |
| heartbeat_at   | the last time the run sent metrics                |
| description    | any notes associated with the run                 |
| system_metrics | the latest system metrics recorded for the run    |

## Special Methods

| Method               | Params                                       | Description                                                                                                                                           |
| -------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| @property<br>summary |                                              | A mutable dict-like property that holds the current summary. Calling update will persist any changes.                                                 |
| history              | _samples=500, stream="default", pandas=True_ | Returns a dataframe containing the number of samples specified captured during the run. If stream is set to "system", returns system metrics instead. |
| files                | _names=[], per_page=50_                      | Returns files associated with this run. If you pass names you limit to only files with those names                                                    |
| file                 | _name_                                       | Returns a specific file.                                                                                                                              |
| update               |                                              | Saves any local changes to the server. Currently supports persisting changes to summary, config, tags, and description                                |

## File Attributes

| Attribute  | Description                            |
| ---------- | -------------------------------------- |
| name       | a list of tags associated with the run |
| url        | the source url                         |
| md5        | and md5 of the content                 |
| mimetype   | the mimetype of the content            |
| updated_at | updated timestamp                      |
| size       | size of the file in bytes              |

## Special Methods

| Method   | Params          | Description                                                                                   |
| -------- | --------------- | --------------------------------------------------------------------------------------------- |
| download | _replace=False_ | Download the source file in the current directory. If replace is True, replace existing files |

