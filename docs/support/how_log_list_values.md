---
title: "How do I log a list of values?"
displayed_sidebar: support
tags:
   - experiments
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- Logging lists directly is not supported. List-like collections of numerical data convert to [histograms](../../../ref/python/data-types/histogram.md). To log entries from a list, assign a name to each entry and use those names as keys in a dictionary, as shown below. -->

<Tabs
  defaultValue="dictionary"
  values={[
    {label: 'Using a dictionary', value: 'dictionary'},
    {label: 'As a histogram', value: 'histogram'},
  ]}>
  <TabItem value="dictionary">

```python
wandb.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
  </TabItem>
  <TabItem value="histogram">

```python
wandb.log({"losses": wandb.Histogram(losses)})  # Converts losses to a histogram
```
  </TabItem>
</Tabs>