---
displayed_sidebar: default
---

# Set up Docker
The following sections outline how to configure a launch queue and agent to execute jobs in Docker. 

## Configure a queue for Docker

Create a queue in the W&B App that uses Docker as its compute resource:

1. Navigate to the [Launch page](https://wandb.ai/launch).
3. Click on the **Create Queue** button.
4. Select the **Entity** you would like to create the queue in.
5. Enter a name for your queue in the **Name** field.
6. Select **Docker** as the **Resource**. 
7. Define your Docker queue configuration in the **Configuration** field. Resource arguments populate docker run arguments. Therefore, you can pass in any arguments available for the [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command. 


Place the values in a list within the **Configuration** field to handle options that can be specified more than once:

```json
{
    "env": [
        "MY_ENV_VAR=value",
        "MY_EXISTING_ENV_VAR"
    ],
    "volume": [
         "/mnt/datasets:/mnt/datasets"
    ]
}
```





8. After you configure your queue, click on the **Create Queue** button.

![](/images/launch/create-queue.gif)


## Configure a launch agent for Docker
TEXT.


## Start your agent
1. Go to the Launch page at [https://wandb.ai/launch](https://wandb.ai/launch). 
2. Select the entity from the **Entity** dropdown that contains your queue.
3. Find your queue and click on **View queue**.
4. Click on **Add an agent**. 
5. A modal will appear. Copy and paste the command to the machine that had the Docker container running. It will look similar to:

```bash
wandb launch-agent -e <entity> -q <queue-name>
```

Replace the values in `<>` with your W&B entity and the name of your queue.

See the [Start an agent](./run-agent.md) for more information on how agents work and for information on optional configuration.

:::info
The launch agent's default behavior is to perform any necessary container builds by running `docker build` on its local host. The agent will execute runs from a Docker queue by running `docker run` on its local host. The agent uses the Docker CLI for these actions, so any Docker CLI configuration that you have set up on your machine will be used by the agent.
:::


## (Optional) Additional queue configurations for Docker

### Specify a GPU for your Docker queue
Add the `gpus` key to your queue resource configuration[LINK] to control which NVIDIA GPU the launch agent uses for executing jobs.

:::note
You will need to install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the machine where your agent is running in order to leverage GPU through Docker.
:::


For example, the following code snippet shows how to [INSERT]:

```json
{
    "gpus": "all"
}
```

For more information, see the relevant [NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration).

:::info
The `gpus` key of the resource configuration is used to pass values to the `--gpus` argument of `docker run`.
:::


<!-- TODO: put this in a technical FAQ or in the queue docs -->
### TensorFlow on GPUs

You might need to specify a custom base image for the container builder for the agent if you have a launch job that will TensorFlow on a GPU. To ensure TensorFlow properly utilizes your GPU, specify a Docker image and its image tag for the `builder.accelerator.base_image` key in the queue resource configuration. 

For example, the following JSON code snippet shows how to specify the  `tensorflow/tensorflow:latest-gpu` base image:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```