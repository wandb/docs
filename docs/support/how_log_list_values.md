---
title: "How do I log a list of values?"
tags:
   - experiments
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- Logging lists directly is not supported. Instead, list-like collections of numerical data are converted to [histograms](../../../ref/python/data-types/histogram.md). To log all of the entries in a list, give a name to each entry in the list and use those names as keys in a dictionary, as below. -->

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
wandb.log({"losses": wandb.Histogram(losses)})  # converts losses to a histogram
```
  </TabItem>
</Tabs>