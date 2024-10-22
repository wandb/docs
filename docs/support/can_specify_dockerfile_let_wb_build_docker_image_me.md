---
title: "Can I specify a Dockerfile and let W&B build a Docker image for me?"
tags:
   - launch
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Yes. This is particularly useful if you have a lot of requirements that do not change often, but you have a codebase that does change often.

:::important
Ensure your Dockerfile is formatted to use mounts. For more information, see [Mounts documentation on the Docker Docs website](https://docs.docker.com/build/guide/mounts/). 
:::

Once your Dockerfile is configured, you can then specify your Dockerfile in one of three ways to W&B:

* Use Dockerfile.wandb
* W&B CLI
* W&B App


<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'W&B CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

Include a file called `Dockerfile.wandb` in the  same directory as the W&B run’s entrypoint.  W&B will use `Dockerfile.wandb` instead of W&B’s built-in Dockerfile.


  </TabItem>
  <TabItem value="cli">

Provide the `--dockerfile` flag when you call queue a launch job with the [`wandb launch`](../ref/cli/wandb-launch.md) command:

```bash
wandb launch --dockerfile path/to/Dockerfile
```


  </TabItem>
  <TabItem value="app">


When you add a job to a queue on the W&B App, provide the path to your Dockerfile in the **Overrides** section. More specifically, provide it as a key-value pair where `"dockerfile"` is the key and the value is the path to your Dockerfile. 

For example, the following JSON shows how to include a Dockerfile that is within a local directory:

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



## Permissions and Resources