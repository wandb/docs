---
title: "InitStartError: Error communicating with wandb process"
displayed_sidebar: support
tags:
   - experiments
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

This error indicates that the library encounters an issue launching the process that synchronizes data to the server.

The following workarounds resolve the issue in specific environments:

<Tabs
  defaultValue="linux"
  values={[
    {label: 'Linux and OS X', value: 'linux'},
    {label: 'Google Colab', value: 'google_colab'},
  ]}>
  <TabItem value="linux">

```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```
  </TabItem>
  <TabItem value="google_colab">

For versions prior to `0.13.0`, use:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
  </TabItem>
</Tabs>