---
slug: /guides/app/features/panels/weave
description: >-
  Some features on this page are in beta, hidden behind a feature flag. Add
  `weave-plot` to your bio on your profile page to unlock all related features.
displayed_sidebar: default
---

# Weave

To learn how to write your own queries interactively, **check out [this report](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr)**, which goes from the basic operations available in Weave to other advanced visualizations of your data.

## Introduction

Weave Panels allow users to directly query W&B for data, visualize the results, and further analyze interactively. Adding Weave Panels is really easy:
* In your **Workspace**, click on `Add Panel` and select `Weave`.
![](/images/weave/add_weave_panel_workspace.png)
* In a **Report**:
  * Type `/weave`and select `Weave` to add an independent Weave Panel.
  ![](/images/weave/add_weave_panel_report_1.png)
  * Type `/Panel grid` -> `Panel grid` and then click on `Add panel` -> `Weave` to add a Weave Panel associated with a set of runs.
  ![](/images/weave/add_weave_panel_report_2.png)

## Components

### Weave Expression

Weave Expressions allow the user to query the data stored in W&B - from runs, to artifacts, to models, to tables, and more! The most common Weave Expression is generated from logging a Table,`wandb.log({"cifar10_sample_table":<MY_TABLE>})`, and will look like this:

![](/images/weave/basic_weave_expression.png)

Let's break this down:

* `runs` is a **variable** automatically injected in Weave Panel Expressions when the Weave Panel is in a Workspace. Its "value" is the list of runs which are visible for that particular Workspace. [Read about the different attributes available within a run here](../../../../track/public-api-guide.md#understanding-the-different-attributes).
* `summary` is an **op** which returns the Summary object for a Run. Note: **ops** are "mapped", meaning this **op** is applied to each Run in the list, resulting in a list of Summary objects.
* `["cifar10_sample_table"]` is a Pick **op** (denoted with brackets), with a **parameter** of "predictions". Since Summary objects act like dictionaries or maps, this operation "picks" the "predictions" field off of each Summary object. As noted above, the "predictions" field is a Table, and therefore this query results in the Table above.

### Weave Configuration

Click on the gear icon on the upper left corner of the panel to expand the Weave Configuration. This allows the user to configure the type of panel and the parameters for the result panel.

![](/images/weave/weave_panel_config.png)

### Weave Result Panel

Finally, the Weave Result Panel renders the result of the Weave Expression, using the selected Weave Panel, configured by the configuration to display the data in an interactive form. Here we can see a Table and a Plot of the same data.

![](/images/weave/result_panel_table.png)

![](/images/weave/result_panel_plot.png)

## Basic Operations

### Sort
You can easily sort from the column options
![](/images/weave/weave_sort.png)

### Filter
You can either filter directly in the query (first image) or using the filter button ▼ in the top left corner (second image)
![](/images/weave/weave_filter_1.png)
![](/images/weave/weave_filter_2.png)

### Map
Map operation just iterates over data and applies a function. This can be done both directly from the Weave query (first image) or by inserting a new column from the column options (second gif)
![](/images/weave/weave_map.png)
![](/images/weave/weave_map.gif)

### Groupby
You can filter using a query (first image) or from the colum options (second gif)
![](/images/weave/weave_groupby.png)
![](/images/weave/weave_groupby.gif)

### Concat
The concat operation allows you to concatenate 2 tables and concatenate or join from the panel settings
![](/images/weave/weave_concat.gif)

### Join
It is also possible to join tables directly in the query, where:
* `project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat` is the first table
* `project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat` is the second table
* `(row) => row["Label"]` are selectors for each table, determining which column to join on
* `"Table1"` and `"Table2"` are the names of each table when joined
* `true` and `false` are for left and right inner/outer join settings
![](/images/weave/weave_join.png)

## Runs Object
Among other things, Weave allows you to access the `runs` object, which stores a detailed record of your experiments. You can find more details about it in [this section](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object) of the report but, as quick overview, `runs` object has available:
* `summary`: A dictionary of information that summarizes the run's results. This can be scalars like accuracy and loss, or large files. By default, `wandb.log()` sets the summary to the final value of a logged time series. The contents of the summary can also be set directly. Think of the summary as the run's "outputs". You are probably familiarized with this if you are using tables, as those are stored under the summary of the run and accesses with an expression like `runs.summary[<table-key>]`.
* `history`: A list of dictionaries meant to store values that change while the model is training such as loss. The command `wandb.log()` appends to this object.
* `config`: A dictionary of the run's configuration information, such as the hyperparameters for a training run or the preprocessing methods for a run that creates a dataset Artifact. Think of these as the run's "inputs"
![](/images/weave/weave_runs_object.png)

## Accessing Artifacts

Artifacts are a core concept in W&B. They are a versioned, named collection of files and directories. Artifacts can be used to track model weights, datasets, and any other file or directory. Artifacts are stored in W&B and can be downloaded or used in other runs. You can find more details and examples in [this section](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts) of the report. Artifacts are normally accessed from the `project` object:
* `project.artifactVersion()`: returns the specific artifact version for a given name and version within a project
* `project.artifact("")`: returns the artifact for a given name within a project. You can then use `.versions` to get a list of all versions of this artifact
`project.artifactType()`: returns the `artifactType` for a given name within a project. You can then use `.artifacts` to get a list of all artifacts with this type
`project.artifactTypes`: returns a list of all artifact types under the project
![](/images/weave/weave_artifacts.png)





