---
title: How do I handle the "Failed to query for notebook" error?
tags:
- notebooks
- environment variables
---
If you're seeing the error message `"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` you can resolve it by setting the environment variable. There's multiple ways to do so:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
  </TabItem>
  <TabItem value="python">

```notebook
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "notebook name here"
```
  </TabItem>
</Tabs>
