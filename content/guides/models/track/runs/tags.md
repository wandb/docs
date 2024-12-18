---
menu:
  default:
    identifier: tags
    parent: what-are-runs
title: Add labels to runs with tags
---

Add tags to label runs with particular features that might not be obvious from the logged metrics or artifact data. 

For example, you can add a tag to a run to indicated that run's model is `in_production`, that run is `preemptible`, this run represents the `baseline`, and so forth.

## Add tags to one or more runs

Programmatically or interactively add tags to your runs.

Based on your use case, select the tab below that best fits your needs:

{{< tabpane text=true >}}
    {{% tab header="W&B Python SDK" %}}
You can add tags to a run when it is created: 

```python
import wandb

run = wandb.init(
  entity="entity",
  project="<project-name>",
  tags=["tag1", "tag2"]
)
```

You can also update the tags after you initialize a run. For example, the proceeding code snippet shows how to update a tag if a particular metrics crosses a pre-defined threshold:

```python
import wandb

run = wandb.init(
  entity="entity", 
  project="capsules", 
  tags=["debug"]
  )

# python logic to train model

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```    
    {{% /tab %}}
    {{% tab header="Public API" %}}
After you create a run, you can update tags using [the Public API](../track/public-api-guide.md). For example:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # you can choose tags based on run data here
run.update()
```    
    {{% /tab %}}
    {{% tab header="Project page" %}}
This method is best suited to tagging large numbers of runs with the same tag or tags.

1. Navigate to your project workspace.
2. Select **Runs** in the from the project sidebar.
3. Select one or more runs from the table.
4. Once you select one or more runs, select the **Tag** button above the table.
5. Type the tag you want to add and select the **Create new tag** checkbox to add the tag.    
    {{% /tab %}}
    {{% tab header="Run page" %}}
This method is best suited to applying a tag or tags to a single run manually.

1. Navigate to your project workspace.
2. Select a run from the list of runs within your project's workspace.
1. Select **Overview** from the project sidebar.
2. Select the gray plus icon (**+**) button next to **Tags**.
3. Type a tag you want to add and select **Add** below the text box to add a new tag.    
    {{% /tab %}}
{{< /tabpane >}}



## Remove tags from one or more runs

Tags can also be removed from runs with the W&B App UI.

{{< tabpane text=true >}}
  {{% tab header="Project page"%}}
This method is best suited to removing tags from a large numbers of runs.

1. In the Run sidebar of the project, select the table icon in the upper-right.  This will expand the sidebar into the full runs table.
2. Hover over a run in the table to see a checkbox on the left or look in the header row for a checkbox to select all runs.
3. Select the checkbox to enable bulk actions. 
4. Select the runs you want to remove tags.
5. Select the **Tag** button above the rows of runs.
6. Select the checkbox next to a tag to remove it from the run.  
  {{% /tab %}}
  {{% tab header="Run page"%}}
1. In the left sidebar of the Run page, select the top **Overview** tab. The tags on the run are visible here.
2. Hover over a tag and select the "x" to remove it from the run.  
  {{% /tab %}}
{{< /tabpane >}}


