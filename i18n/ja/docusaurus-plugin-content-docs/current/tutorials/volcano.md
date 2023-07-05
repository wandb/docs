# Launch multinode jobs with Volcano

This tutorial will guide you through the process of launching multinode training jobs with W&B and Volcano on Kubernetes.

## Overview

In this tutorial, you will learn how to use W&B Launch to run multinode jobs on Kubernetes. The steps we will follow are:

- Ensure that you have a Weights & Biases account and a Kubernetes cluster.
- Create a launch queue for our volcano jobs.
- Deploy a Launch agent into our kubernetes cluster.
- Create a distributed training job.
- Launch our distributed training.

## Prerequisites

Before you get started, you will need:

- A Weights & Biases account
- A Kubernetes cluster

## Create a launch queue

The first step is to create a launch queue. Head to [wandb.ai/launch](https://wandb.ai/launch) and in the top right corner of your screen, hit the blue **Create a queue** button. A queue creation drawer will slide out from the right side of your screen. Select an entity, enter a name, and select **Kubernetes** as the type for your queue.

In the configuration section, we will enter a [volcano job](https://volcano.sh/en/docs/vcjob/) template. Any runs launched from this queue will be created using this job specification, so you can modify this configuration as needed to customize your jobs.

This configuration block can accept a Kubernetes job specification, volcano job specification, or any other custom resource definition (CRD) that you are interested in launching. You can make use of [macros in the configuration block](../guides/launch/kubernetes#custom-controllers) to dynamically set the contents of this spec.

In this tutorial, we will use a configuration for multinode pytorch training that makes use of [volcano's pytorch plugin](https://github.com/volcano-sh/volcano/blob/master/docs/user-guide/how_to_use_pytorch_plugin.md). You can copy and paste the following config:

```json
{
  "kind": "Job",
  "spec": {
    "tasks": [
      {
        "name": "master",
        "policies": [
          {
            "event": "TaskCompleted",
            "action": "CompleteJob"
          }
        ],
        "replicas": 1,
        "template": {
          "spec": {
            "containers": [
              {
                "name": "master",
                "image": "${image_uri}",
                "imagePullPolicy": "IfNotPresent"
              }
            ],
            "restartPolicy": "OnFailure"
          }
        }
      },
      {
        "name": "worker",
        "replicas": 3,
        "template": {
          "spec": {
            "containers": [
              {
                "name": "worker",
                "image": "${image_uri}",
                "workingDir": "/home",
                "imagePullPolicy": "IfNotPresent"
              }
            ],
            "restartPolicy": "OnFailure"
          }
        }
      }
    ],
    "plugins": {
      "pytorch": [
        "--master=master",
        "--worker=worker",
        "--port=23456"
      ]
    },
    "minAvailable": 1,
    "schedulerName": "volcano"
  },
  "metadata": {
    "name": "wandb-job-${run_id}",
    "labels": {
      "wandb_entity": "${entity_name}",
      "wandb_project": "${project_name}"
    }
  },
  "apiVersion": "batch.volcano.sh/v1alpha1"
}
```

Click the **Create queue** button at the bottom of the drawer to finish creating your queue.

## Deploy your launch agent

Now that you have created a queue, you will need to deploy a launch agent to pull and execute jobs from the queue. The easiest way to do this is with the [`launch-agent` chart from W&B's official `helm-charts` repository](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent).

### Get an API key for the agent

If you created a queue in your personal entity, head to [wandb.ai/authorize](https://wandb.ai/authorize) to get a personal API key.

If you created a queue for a W&B, you will need to create a service account in that team or use an API key from a prior service account. For more information on generating service accounts, see [these docs](../guides/track/environment-variables.md#automated-runs-and-service-accounts).

Keep the API key you plan to deploy your agent with handy for later steps.

### Create a config for the agent

Create a yaml configuration file for the agent you will deploy. Refer [here](../guides/launch/run-agent.md#agent-configuration) for complete documentation of the agent configuration file. The config should contain (at least):

```yaml
entity: <your-entity>
queues: [ <your-queue> ]
```

### Deploy with helm

Make sure you have `helm` and `kubectl` installed and pointing at the cluster where you want to install your agent. To add the `wandb/helm-charts` repository, run:

```bash
helm repo add wandb https://wandb.github.io/helm-charts
```

Next, run the following command (substituting your own api key and agent config file):

```bash
helm install volcano-agent wandb/launch-agent --set agent.apiKey=<your-api-key> --set-file launchConfig=<path-to-your-launch-config>
```

If the commands above succeed, you should see the status of the queue you created change from **Not running** to **Active**.

## Create a training job

Volcano's pytorch plugin will automatically configure the necessary environment variables for pytorch ddp to work, e.g. `MASTER_ADDR`, `RANK`, `WORLD_SIZE`, etc. As long as your write your pytorch code uses DDP correctly, everything else should **just work**. Refer to [pytorch's documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details on how to use DDP in your custom python code.

:::tip
Volcano's pytorch plugin is also compatible with [multinode training via the PyTorch Lightning `Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-nodes).
:::

## Launch 🚀

Now that our queue and cluster are set up, it's time to launch some distributed training! To start off with we will use [a job](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) trains a simple multi-layer perceptron on random data using volcano's pytorch plugin. You can find the source code for the job [here](https://github.com/wandb/launch-jobs/tree/main/jobs/distributed_test).

To launch this job, head to the [job's page](https://wandb.ai/wandb/multinodetest/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjc3MDcwNTg1/runs/latest) and click the **Launch** button in the top right corner of the screen. You will be prompted to select a queue to launch the job from.

![](/images/launch/launching_multinode_job.png)

1. Set the jobs parameters however you like,
2. Select the queue you created earlier.
3. Modify the volcano job in the **Resource config** section to modify the parameters of your job. For example, you can change the number of workers by changing the `replicas` field in the `worker` task.
4. Click **Launch** 🚀

You can monitor the progress and if necessary stop your job from the W&B UI.
