---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for SageMaker

Use W&B Launch to send your runs to AWS SageMaker. There are two ways to use Launch on SageMaker:

1. Bring your own image (BYOI) and push it to your Amazon ECR repository. 
2. Let the W&B Launch agent build a container for your and push it to your ECR repository.

:::info
If you bring your own image (1), it must already be SageMaker compatible. If you let W&B build and push the image for you (2), W&B will make your image SageMaker compatible.
:::

The following table highlights the key differences between the two workflows listed above:

|       | BYOI  | Default W&B Launch  |
| ----- | ----- | ----- |
| Allowed job type            | Image sourced-job                                   | Git or code artifact sourced job                      | 
| Queue configuration options | Same for both workflows                            | Same for both workflows                               |
| Agent configuration options |                 N/A                                | Must have the `registry` block in your agent config file |
| Builder options             |         Docker, Kaniko, Noop                       | Docker, Kaniko |


:::tip
**Should I bring my own image or let W&B build and push the image for me?**

Follow the bring your own image (BYOI) method if you uses packages that are not available on PyPi or find that you are having issues with W&B building your images. 
In which case, we appreciate feedback about the specific problem so we can fix it in future releases.
:::



## Prerequisites
Create and make note of the following AWS resources:

1. **Setup SageMaker in your AWS account.** See the [SageMaker Developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html) for more information. 
2. **Create an IAM execution role.** See the [INSERT] section to view the JSON policy to add to your IAM role. 
3. **Create an Amazon ECR repository**  to store images you want to execute on SageMaker. See the [Amazon ECR documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for more information.
4. **(If W&B creates your image) Create an Amazon S3 bucket** to store SageMaker outputs from your runs. See the [Amazon S3 documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) for more information. 

Ensure that the AWS account associated with the launch agent has access to the Amazon ECR repo, Amazon S3 bucket and the ability to create Sagemaker jobs. 


### IAM Role policy

Text.


## Configure a queue for SageMaker
Create a queue in the W&B App that uses SageMaker as its compute resource:

1. Navigate to the [Launch App](https://wandb.ai/launch).
3. Click on the **Create Queue** button.
4. Select the **Entity** you would like to create the queue in.
5. Provide a name for your queue in the **Name** field.
6. Select **SageMaker** as the **Resource**.
7. Within the **Configuration** field, provide information about your SageMaker job. By default, W&B will populate a YAML and JSON CreateTrainingJob request body:

<Tabs
  defaultValue="JSON"
  values={[
    {label: 'JSON', value: 'JSON'},
    {label: 'YAML', value: 'YAML'},
  ]}>
  <TabItem value="JSON">

```json title='Queue configuration'
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

  </TabItem>
  <TabItem value="YAML">

```yaml title='Queue configuration'
RoleArn: <REQUIRED>
ResourceConfig:
  InstanceType: ml.m4.xlarge
  InstanceCount: 1
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: <REQUIRED>
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

  </TabItem>
</Tabs>

Specify your:

- `RoleArn` : ARN of the IAM role that you created that satisfied the prerequisites. [LINK]
- `OutputDataConfig.S3OutputPath` : An Amazon S3 URI specifying where SageMaker outputs will be stored. 

:::tip
The launch queue configuration for a SageMaker compute resource is passed to the [`CreateTrainingJob` SageMaker API request](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html). This means that you can optionally add additional arguments to your launch queue configuration that correspond to SageMaker CreateTrainingJob request parameters.
:::


7. After you configure your queue, click on the **Create Queue** button.

## Configure a launch agent for SageMaker

Configure a launch agent to execute jobs from your queues with SageMaker. The following steps outline how to configure your launch agent to use SageMaker with Launch. 

1. **Set the AWS credentials** you want the agent to use either by: 
    * Set [AWS SDK for Python environment variables](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables) 
    or
    * Set a `default` profile in your [AWS config](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#shared-credentials-file)(`~/.aws/config`). 
    
    The following code snippet shows an example of how to set your AWS config `~/.aws/config` file. Replace the `<>` with your own values:

    ```yaml title="~/.aws/config"
    [default]
    aws_access_key_id=<your_aws_access_key_id>
    aws_secret_access_key=<your_aws_secret_access_key>
    aws_session_token=<your_aws_session_token>
    ```

    For more information about AWS CLI credentials, see the [Configure the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) documentation for more information.

2. **Define the agent config**. Add the `environment` block in your agent config file (`~/.config/wandb/launch-config.yaml`). 

    The following code snippet shows an example `launch-config.yaml` file. Ensure you specify the type as AWS and the region of your Amazon S3 bucket and ECR repository:

    ```yaml title="~/.config/wandb/launch-config.yaml"
    environment:
        type: aws
        region: <aws-region>  # E.g. us-east-2
    ```

:::note
Continue to complete the following steps if you want W&B to build and push your image for you. 

Skip to the [Start your agent](#start-your-agent) section if you brought your own image and you pushed that image to your ECR repository.
:::


3. **(Optional) Specify a `registry`**: If you want the W&B agent to build new containers and push them to ECR for you, you will need to add a `registry` block to your agent config.

    ```yaml title="~/.config/wandb/launch-config.yaml"
    registry:
        type: ecr
        uri: <ecr-repo-uri>
    ```

4. **(Optional) Enable Kaniko**
    If you run the agent in Kubernetes you can enable Kaniko builds by adding the following to you agent config:

    ```yaml title="~/.config/wandb/launch-config.yaml"
    builder:
        type: kaniko
        build-context-store: s3://<s3-bucket>/<prefix>
    ```

    Kaniko will store compressed build contexts in the local specified under `build-context-store` and then push any container images it builds to the ECR repository configured in the `registry` block. Kaniko pods will need permission to access the S3 bucket specified in `build-context-store` and read/write access to the ECR repository specified in `registry.repository`.



## Start your agent
Run the agent locally, in a Kubernetes cluster, or in a Docker container.  The launch agent will continuously run launch jobs on Amazon SageMaker so long as the agent is an environment with AWS credentials.


Copy and paste the following. Ensure to replace values in `<>` with your own values:


```bash
wandb launch-agent -e <your-entity> -q <queue-name>  \\ 
    -c <path-to-agent-config>
```


Another common pattern is to run the agent on an Amazon EC2 instance. The agent can perform container builds and push them to Amazon ECR if you install Docker on an Amazon Linux 2 instance. The launch agent can then launch jobs on SageMaker using the AWS credentials associated with the EC2 instance. AWS provides a guide to installing Docker in Amazon Linux 2 [here](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html#prequisites).






## (Optional) Additional permissions for Kaniko builder
Add the following permissions to the IAM Role you specify in your [SageMaker queue configuration](#configure-a-queue-for-sagemaker) if you use Kaniko to build containers with Amazon SageMaker.

The launch agent requires access to Amazon ECR. More specifically, the launch agent requires permission to push and pull container images and access to an Amazon S3 bucket.


Replace the  values in `<PLACEHOLDER>` with your own values:

```json title='IAM role policy'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:PutLifecyclePolicy",
        "ecr:PutImageTagMutability",
        "ecr:StartImageScan",
        "ecr:CreateRepository",
        "ecr:PutImageScanningConfiguration",
        "ecr:UploadLayerPart",
        "ecr:BatchDeleteImage",
        "ecr:DeleteLifecyclePolicy",
        "ecr:DeleteRepository",
        "ecr:PutImage",
        "ecr:CompleteLayerUpload",
        "ecr:StartLifecyclePolicyPreview",
        "ecr:InitiateLayerUpload",
        "ecr:DeleteRepositoryPolicy"
      ],
      "Resource": "arn:aws:ecr:<REGION>:<ACCOUNT-ID>:repository/<YOUR-REPO-NAME>"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:BatchCheckLayerAvailability",
      "Resource": "arn:aws:ecr:<REGION>:<ACCOUNT-ID>:repository/<YOUR-REPO-NAME>"
    },
    {
      "Sid": "ListObjectsInBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>"]
    },
    {
      "Sid": "AllObjectActions",
      "Effect": "Allow",
      "Action": "s3:*Object",
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>/*"]
    }
  ]
}
```
<!-- 
In order to use SageMaker queues, you will also need to create a separate execution role that is assumed by your jobs running in SageMaker. The agent should be granted the following permissions to be allowed to create training jobs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:CreateTrainingJob",
      "Resource": "arn:aws:sagemaker:<REGION>:<ACCOUNT-ID>/*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "<ARN-OF-ROLE-TO-PASS>"
    },
    {
      "Effect": "Allow",
      "Action": "kms:CreateGrant",
      "Resource": "<ARN-OF-KMS-KEY>",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "sagemaker.<REGION>.amazonaws.com",
          "kms:GrantIsForAWSResource": "true"
        }
      }
    }
  ]
}
``` -->

:::note
The `kms:CreateGrant` permission for SageMaker queues is required only if the associated ResourceConfig has a specified VolumeKmsKeyId and the associated role does not have a policy that permits this action.
:::


