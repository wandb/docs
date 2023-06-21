---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Launch on SageMaker

Use W&B Launch to send your runs to AWS SageMaker. There are two ways to use Launch on SageaMaker:

1. Bring your own image (BYOI) and push it to your Amazon ECR repository. 
2. Let the agent build a container for your and push it to your ECR repository.

:::info
If you bring your own image (1), it must already be SageMaker compatible. If you let W&B build and push the image for you (2), W&B will make your image SageMaker compatible.
:::

The following table highlights the key differences between the two workflows:

|       | BYOI  | W&B   |
| ----- | ----- | ----- |
| Queue | Same for both workflows | Same for both workflows. |
| Job type                  | Image source-job | Git or code artifact sourced job | 
| Job name | Name must match the name of image in ECR repo | Job name does not need to match. |  
| Agent configuration |  | Must have the `registry` block in your agent config file |


:::tip
**Should I bring my own image or let W&B build and push the image for me?**

[INSERT] Something about if the W&B default build system does not satisfy the customer's needs.
:::

## Prerequisites
Create the following AWS resources to launch your W&B runs on AWS SageMaker:

1. **Setup SageMaker in your AWS account.** See the [SageMaker Developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html) for more information.
2. **Create an IAM execution role.** Attach the [AmazonSageMakerFullAccess policy to your role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
3. **Create an Amazon ECR repository**  to store images you want to execute on SageMaker. See the [Amazon ECR documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for more information.
4. **Create an Amazon S3 bucket** to store SageMaker outputs from your runs. See the [Amazon S3 documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) for more information.

Make note of your IAM RoleARN, your Amazon S3 URI, and your ECR repository name.

## 1. Create a queue
Create a queue in the W&B App that uses SageMaker as its compute resource. When you create a queue, a config is automatically populated with a template of key-value pairs. (see below)

Within the queue config, you must provide:

- `RoleArn` : ARN of the role the IAM role that will be assumed by the job.
- `OutputDataConfig.S3OutputPath` : An S3 URI specifying where SageMaker outputs will be stored.

Follow these steps to create a queue: 

1. Navigate to the [Launch application](https://wandb.ai/launch).
3. Click on the **Create Queue** button.
4. Select the **entity** you would like to create the queue in.
5. Enter a name for your queue in the **Name** field.
6. Select **SageMaker** as the **Resource**.
7. Enter a configuration for your queue. The config is a JSON blob that:


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

You can optionally add additional arguments.  See [INSERT] for more information.

8. Click on the **Create Queue** button.





## 2. Configure the launch agent

Edit your AWS credentials file and your W&B config file to configure your agent.  

:::note
The agent can build container images with either Kaniko or Docker. The agent will attempt to perform container builds with `docker build` by default. 
:::

<!-- ### 1. Define AWS credentials -->

The following steps outline the necessary steps to configure your agent to use SageMaker with Launch, for more information on how launch agent work, see the Start an agent[LINK] page.

1. **Set the AWS credentials** you want the agent to use with either: 
    * [AWS SDK for Python environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables).
    * Set a `default` profile in your [AWS config](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file)(`~/.aws/config`). 
    
    The following code snippet shows an example `~/.aws/config` file:

    ```yaml title="~/.aws/config"
    [default]
    aws_access_key_id=<your_aws_access_key_id>
    aws_secret_access_key=<your_aws_secret_access_key>
    aws_session_token=<your_aws_session_token>
    ```

2. **Define the agent config**. Add the `environment` block in your agent config file (`~/.config/wandb/launch-config.yaml`). The following code snippet shows an example `launch-config.yaml` file:

    ```yaml title="~/.config/wandb/launch-config.yaml"
    environment:
        type: aws
        region: <aws-region>  # E.g. us-east-2
    ```

:::note
Continue to complete the following steps if you want W&B to build and push your image for you. If you bring your own image, skip the next two steps and move on to the [Add jobs to your queue section](#3-add-jobs-to-your-queue).
:::


3. **(Optional) Specify a `registry`**: If you want the W&B agent to build new containers and push them to ECR for you, you will need to add a `registry` block to your agent config.

    ```yaml title="~/.config/wandb/launch-config.yaml"
    registry:
        type: ecr
        repository: <ecr-repo-name>
    ```

4. **(Optional) Enable Kaniko**
    If you run the agent in Kubernetes you can enable Kaniko builds by adding the following to you agent config:

    ```yaml title="~/.config/wandb/launch-config.yaml"
    builder:
        type: kaniko
        build-context-store: s3://<s3-bucket>/<prefix>
    ```

    Kaniko will store compressed build contexts in the local specified under `build-context-store` and then push any container images it builds to the ECR repository configured in the `registry` block. Kaniko pods will need permission to access the S3 bucket specified in `build-context-store` and read/write access to the ECR repository specified in `registry.repository`.



## 3. Add jobs to your queue


[TO DO - Elaborate more/quickly reiterate what was said in table above,]

## 4. Start your agent
Run the agent locally, in a Kubernetes cluster, or in a Docker container.  The launch agent will continuously run launch jobs on Amazon SageMaker so long as the agent is an environment with AWS credentials.


Copy and paste the following. Ensure to replace values in `<>` with your own values:


```bash
wandb launch-agent -e <your-entity> -q <queue-name>  \\ 
    -c <path-to-agent-config>
```

For more information on launch agents, see the Start an agent[LINK] page.







Another common pattern is to run the agent on an Amazon EC2 instance. The agent can perform container builds and push them to Amazon ECR if you install Docker on an Amazon Linux 2 instance. The launch agent can then launch jobs on SageMaker using the AWS credentials associated with the EC2 instance. AWS provides a guide to installing Docker in Amazon Linux 2 [here](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html#prequisites).


