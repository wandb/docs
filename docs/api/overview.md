---
title: API Overview
sidebar_label: Overview
---

W&B provides an API to import and export data directly. This is useful for doing custom analysis of your existing runs or running an evaluation script and adding additional summary metrics.

## Authentication

Before using the API you need to store your key locally by running `wandb login` or set the **WANDB_API_KEY** environment variable.

## Single Run Example

This script finds all the metrics saved for a single run and saves them to a csv

```python
import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("oreilly-class/cifar/uxte44z7")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")
```

## Project Example

This script finds a project and outputs a csv of runs with name, configs and summary stats. 

```python
import wandb
api - wandb.Api()

# Change oreilly-class/cifar to <entity/project-name>
runs = api.runs("oreilly-class/cifar")
summary_list = [] 
config_list = [] 
name_list = [] 
for run in runs: 
    # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict) 

    # run.config is the input metrics.  We remove special values that start with _.
    config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 
    
    # run.name is the name of the run.
    name_list.append(run.name)       

import pandas as pd 
summary_df = pd.DataFrame.from_records(summary_list) 
config_df = pd.DataFrame.from_records(config_list) 
name_df = pd.DataFrame({'name': name_list}) 
all_df = pd.concat([name_df, config_df,summary_df], axis=1)

all_df.to_csv("project.csv")

```

## Error handling

If errors occur while talking to W&B servers a `wandb.CommError` will be raised. The original exception can be introspected via the **exc** attribute.
