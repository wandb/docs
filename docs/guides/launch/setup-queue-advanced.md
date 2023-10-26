---
displayed_sidebar: default
---

# Advanced queue set up

Queue configs can be dynamically configured using macros that are evaluated when the agent dequeues a job from the queue. You can set the following macros:

| Macro             | Description                                           |
|-------------------|-------------------------------------------------------|
| `${project_name}` | The name of the project the run is being launched to. |
| `${entity_name}`  | The owner of the project the run being launched to.   |
| `${run_id}`       | The id of the run being launched.                     |
| `${run_name}`     | The name of the run that is launching.                |
| `${image_uri}`    | The URI of the container image for this run.          |

:::info
Any custom macro not listed in the preceding table (for example `${MY_ENV_VAR}`), is substituted with an environment variable from the agent's environment.
:::

## Use the launch agent to build images that execute on accelerators (GPUs)
You might need to specify an accelerator base image if you use launch to build images that are executed in an accelerator environment.

This accelerator base image must satisfy the following requirements:

- Debian compatibility (the Launch Dockerfile uses apt-get to fetch python)
- Compatibility CPU & GPU hardware instruction set (Make sure your CUDA version is supported by the GPU you intend on using)
- Compatibility between the accelerator version you provide and the packages installed in your ML algorithm
- Packages installed that require extra steps for setting up compatibility with hardware

### How to use GPUs with TensorFlow

Ensure TensorFlow properly utilizes your GPU. To accomplish this, specify a Docker image and its image tag for the `builder.accelerator.base_image` key in the queue resource configuration.

For example, the `tensorflow/tensorflow:latest-gpu` base image ensures TensorFlow properly uses your GPU. This can be configured using the resource configuration in the queue.

The following JSON snippet demonstrates how to specify the TensorFlow base image in your queue config:

```json title="Queue config"
{
    ... rest of queue configuration
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```