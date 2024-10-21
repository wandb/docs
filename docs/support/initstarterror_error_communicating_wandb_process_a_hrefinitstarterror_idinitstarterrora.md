---
title: "`InitStartError: Error communicating with wandb process` <a href="#init-start-error" id="init-start-error"></a>"
tags: []
---

### `InitStartError: Error communicating with wandb process` <a href="#init-start-error" id="init-start-error"></a>
This error indicates that the library is having difficulty launching the process which synchronizes data to the server.

The following workarounds can help resolve the issue in certain environments:

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

For versions prior to `0.13.0` we suggest using:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
  </TabItem>
</Tabs>