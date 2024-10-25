---
title: "How do I handle the 'Failed to query for notebook' error?"
displayed_sidebar: support
tags:
- notebooks
- environment variables
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

If you encounter the error message `"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` resolve it by setting the environment variable. Multiple methods accomplish this:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'Python', value: 'python'},
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