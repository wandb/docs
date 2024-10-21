---
title: "Is there an anaconda package?"
tags: []
---

### Is there an anaconda package?
Yes! You can either install with `pip` or with `conda`. For the latter, you'll need to get the package from the [conda-forge](https://conda-forge.org) channel.

<Tabs
  defaultValue="pip"
  values={[
    {label: 'pip', value: 'pip'},
    {label: 'conda', value: 'conda'},
  ]}>
  <TabItem value="pip">

```bash
# Create a conda env
conda create -n wandb-env python=3.8 anaconda
# Activate created env
conda activate wandb-env
# install wandb with pip in this conda env
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


If you run into issues with this install, please let us know. This Anaconda [doc on managing packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) has some helpful guidance.