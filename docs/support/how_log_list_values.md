---
title: "How do I log a list of values?"
tags:
   - experiments
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


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