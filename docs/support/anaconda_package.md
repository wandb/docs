---
title: "Is there an anaconda package?"
displayed_sidebar: support
tags:
   - other
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

You can install the package using either `pip` or `conda`. For `conda`, obtain the package from the [conda-forge](https://conda-forge.org) channel.

<Tabs
  defaultValue="pip"
  values={[
    {label: 'pip', value: 'pip'},
    {label: 'conda', value: 'conda'},
  ]}>
  <TabItem value="pip">

```bash
# Create a conda environment
conda create -n wandb-env python=3.8 anaconda
# Activate the environment
conda activate wandb-env
# Install wandb using pip
pip install wandb
```

  </TabItem>
  <TabItem value="conda">

```
conda activate myenv
conda install wandb --channel conda-forge
```

  </TabItem>
</Tabs>

For installation issues, refer to this Anaconda [documentation on managing packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) for assistance.