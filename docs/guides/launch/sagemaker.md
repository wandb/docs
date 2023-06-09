---
displayed_sidebar: default
---

# Launch on SageMaker

Use W&B Launch to send your runs to AWS SageMaker.

## Prerequisites
Create the following AWS resources to launch your W&B runs on AWS SageMaker:

1. **Setup SageMaker in your AWS account.** See the [SageMaker Developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html) for more information.
2. **Create an IAM execution role.** Attach the [AmazonSageMakerFullAccess policy to your role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
3. **Create an Amazon ECR repository**  to store images you want to execute on SageMaker. See the [Amazon ECR documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for more information. You can use launch to run custom images you add to this repository yourself or configure the agent to build containers for you and store them in this repository.
4. **Create an Amazon S3 bucket** to store SageMaker outputs from your runs. See the [Amazon S3 documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) for more information.

## Create a queue

Before you can launch a job on SageMaker, you need to create a SageMaker queue in the W&B App. To create a SageMaker queue:

1. Navigate to the [Launch application](https://wandb.ai/launch).
2. Click on the **Queues** tab.
3. Click on the **Create Queue** button.
4. Select the **entity** you would like to create the queue in.
5. Enter a name for your queue.
6. Select **SageMaker** as the **Resource**.
7. Enter a configuration for your queue.
8. Click on the **Create Queue** button.


### Queue configuration

The config for a SageMaker queue is a JSON blob that is passed to [SageMaker API's `CreateTrainingJob` request](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html). When you create the queue, the config will be automatically populated with the following JSON:

```yaml
{
    "RoleArn": "<REQUIRED>",
    "ResourceConfig": {
        "InstanceType": "ml.m4.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 2
    },
    "OutputDataConfig": {
        "S3OutputPath": "<REQUIRED>"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 3600
    }
}
```

You can optionally add additional arguments. However, you must specify:

- `RoleArn` : ARN of the role the IAM role that will be assumed by the job.
- `OutputDataConfig.S3OutputPath` : An S3 URI specifying where SageMaker outputs will be stored.


## Configure and deploy an agent


### Agent configuration
In order for the launch agent to launch jobs with SageMaker you will need configure AWS credentials for you agent. Set the AWS credentials you want the agent to use via [environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables) or as the `default` profile in your [AWS config](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file) and then add an `environment` block to your agent config.


```yaml
environment:
	type: aws
	region: <aws-region>  # E.g. us-east-2
```

Any containers you want to run in SageMaker will need to be stored in your Elastic Container Registry (ECR). If you want your agent to build new containers and push them to ECR for you, you will need to add a `registry` block to your agent config.

```yaml
registry:
	type: ecr
	repository: <ecr-repo-name>
```

The agent can build container images with either Kaniko or Docker. The agent will attempt to perform container builds with `docker build` by default. 

If you run the agent in Kubernetes you can enable Kaniko builds by adding the following to you agent config:

```yaml
builder:
	type: kaniko
	build-context-store: s3://<s3-bucket>/<prefix>
```

Kaniko will store compressed build contexts in the local specified under `build-context-store` and then push any container images it builds to the ECR repository configured in the `registry` block. Kaniko pods will need permission to access the S3 bucket specified in `build-context-store` and read/write access to the ECR repository specified in `registry.repository`.

### Deploy the agent
Run the agent locally, in a Kubernetes cluster, or in a Docker container. The launch agent will continuously run launch jobs on Amazon SageMaker so long as the agent is an environment with AWS credentials.

For more information on deploying an agent to a Kubernetes cluster, see the [Kubernetes deployment guide](./kubernetes.md#deploying-an-agent).

Another common pattern is to run the agent on an Amazon EC2 instance. The agent can perform container builds and push them to Amazon ECR if you install Docker on an Amazon Linux 2 instance. The launch agent can then launch jobs on SageMaker using the AWS credentials associated with the EC2 instance. AWS provides a guide to installing Docker in Amazon Linux 2 [here](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html#prequisites).