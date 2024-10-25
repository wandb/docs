---
title: "Can I specify a Dockerfile and let W&B build a Docker image for me?"
displayed_sidebar: support
tags:
   - launch
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

This feature suits projects with stable requirements but frequently changing codebases.

:::important
Format your Dockerfile to use mounts. For further details, visit the [Mounts documentation on the Docker Docs website](https://docs.docker.com/build/guide/mounts/).
:::

After configuring the Dockerfile, specify it in one of three ways to W&B:

* Use Dockerfile.wandb
* Use W&B CLI
* Use W&B App

<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'W&B CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

Include a `Dockerfile.wandb` file in the same directory as the W&B run's entrypoint. W&B utilizes this file instead of the built-in Dockerfile. 

  </TabItem>
  <TabItem value="cli">

Use the `--dockerfile` flag with the `wandb launch` command to queue a job:

```bash
wandb launch --dockerfile path/to/Dockerfile
```

  </TabItem>
  <TabItem value="app">

When adding a job to a queue in the W&B App, provide the Dockerfile path in the **Overrides** section. Enter it as a key-value pair with `"dockerfile"` as the key and the path to the Dockerfile as the value.

The following JSON demonstrates how to include a Dockerfile in a local directory:

```json title="Launch job W&B App"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```

  </TabItem>
</Tabs>

