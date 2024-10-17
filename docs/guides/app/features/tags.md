---
displayed_sidebar: default
title: Add tags to runs
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Add tags to label runs with particular features that might not be obvious from the logged metrics or artifact data. For example, you can add a tag to a run to indicated that that run's model is `in_production`, that run is `preemptible`, this run represents the `baseline`, and so forth.

## How to add tags

You can add tags to a run when it is created: `wandb.init(tags=["tag1", "tag2"])` .

You can also update the tags of a run during training (for example, if a particular metrics crosses a pre-defined threshold):

```python
run = wandb.init(entity="entity", project="capsules", tags=["debug"])

# python logic to train model

if current_loss < threshold:
    run.tags = run.tags + ("release_candidate",)
```

There are also several ways to add tags after runs have been logged to W&B.

<Tabs
  defaultValue="publicapi"
  values={[
    {label: 'Using the Public API', value: 'publicapi'},
    {label: 'Project Page', value: 'projectpage'},
    {label: 'Run Page', value: 'runpage'},
  ]}>
  <TabItem value="publicapi">

After you create a run, you can update tags using [the Public API](../../../guides/track/public-api-guide.md). For example:

```python
run = wandb.Api().run("{entity}/{project}/{run-id}")
run.tags.append("tag1")  # you can choose tags based on run data here
run.update()
```

Read more about how to use the Public API in the [reference documentation](../../../ref/README.md) or [guide](../../../guides/track/public-api-guide.md).

  </TabItem>
  <TabItem value="projectpage">

This method is best suited to tagging large numbers of runs with the same tag or tags.

1. In the [runs sidebar](../pages/project-page.md#search-for-runs) of the [Project Page](../pages/project-page.md), select the table icon in the upper-right.  This expands the sidebar into the full [runs table](runs-table.md).
2. Hover your mouse over a run in the table to see a checkbox on the left or look in the header row for a checkbox that select all runs.
3. Select the checkbox to enable bulk actions. 
4. Select the runs to which you want to apply your tags.
5. Select the **Tag** button above the rows of runs.
6. Type the tag you want to add and select the **Create new tag** checkbox to add the tag.

  </TabItem>
  <TabItem value="runpage">

This method is best suited to applying a tag or tags to a single run manually.

1. In the left sidebar of the [Run Page](../pages/run-page.md), select the top [Overview tab](../pages/run-page.md#overview-tab).
2. Select the gray plus icon (**+**) button next to **Tags**.
3. Type a tag you want to add and select **Add** below the text box to add a new tag.

  </TabItem>
</Tabs>



## How to remove tags

Tags can also be removed from runs with the W&B App UI.

<Tabs
  defaultValue="projectpage"
  values={[
    {label: 'Project Page', value: 'projectpage'},
    {label: 'Run Page', value: 'runpage'},
  ]}>
  <TabItem value="projectpage">

This method is best suited to removing tags from a large numbers of runs.

1. In the [runs sidebar](../pages/project-page.md#search-for-runs) of the [Project Page](../pages/project-page.md),  select the table icon in the upper-right.  This will expand the sidebar into the full [runs table](runs-table.md).
2. Hover over a run in the table to see a checkbox on the left or look in the header row for a checkbox to select all runs.
3. Select the checkbox to enable bulk actions. 
4. Select the runs you want to remove tags.
5. Select the **Tag** button above the rows of runs.
6. Select the checkbox next to a tag to remove it from the run.

  </TabItem>
  <TabItem value="runpage">

1. In the left sidebar of the [Run Page,](../pages/run-page.md) select the top [Overview tab](../pages/run-page.md#overview-tab). The tags on the run are visible here.
2. Hover over a tag and select the "x" to remove it from the run.

  </TabItem>
</Tabs>
