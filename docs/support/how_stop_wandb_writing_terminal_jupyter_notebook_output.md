---
title: "How do I stop wandb from writing to my terminal or my jupyter notebook output?"
tags:
   - None
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Set the environment variable [`WANDB_SILENT`](../guides/track/environment-variables.md) to `true`.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Jupyter Notebook', value: 'notebook'},
    {label: 'Command Line', value: 'command-line'},
  ]}>
  <TabItem value="python">

```python
os.environ["WANDB_SILENT"] = "true"
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="command-line">

```python
WANDB_SILENT=true
```

  </TabItem>
</Tabs>