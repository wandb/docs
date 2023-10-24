---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for SageMaker

<!-- You can use W&B Launch to submit jobs to run as SageMaker training jobs. Amazon SageMaker training jobs allow users to train machine learning models using provided or custom algorithms on the SageMaker platform. Once initiated, SageMaker handles the underlying infrastructure, scaling, and orchestration. -->

You can use W&B Launch to submit launch jobs to Amazon SageMaker to train machine learning models using provided or custom algorithms on the SageMaker platform.

Launch jobs sent to Amazon SageMaker are executed as SageMaker Training Jobs with the [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html). Arguments to the `CreateTrainingJob` API are controlled with the launch queue configuration. 


Amazon SageMaker [uses Docker images to execute training jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html). Images pulled by SageMaker must be stored in the Amazon Elastic Container Registry (ECR). This means that the image you use for training must be stored on ECR. To learn more about how to set up Launch with ECR, see [Advanced agent set up](./setup-agent-advanced.md).

Amazon SageMaker requires an IAM execution role. The IAM role is used within the SageMaker training job instances to control access to required resources like ECR and Amazon S3. Make note of the IAM role ARN. You will need to specify the IAM role ARN in your queue configuration. 


## Prerequisites
Create and make note of the following AWS resources:

1. **Setup SageMaker in your AWS account.** See the [SageMaker Developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html) for more information.
2. **Create an Amazon ECR repository**  to store images you want to execute on Amazon SageMaker. See the [Amazon ECR documentation](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) for more information.
3. **Create an Amazon S3 bucket(s)** to store SageMaker inputs and outputs for your SageMaker training jobs. See the [Amazon S3 documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) for more information. Make note of the S3 bucket URI and directory.
4. **Create IAM execution role.** The role used in the SageMaker training job requires the following permissions to work. These permissions allow for logging events, pulling from ECR, and interacting with input and output buckets. 
  ```json title="IAM role policy"
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "cloudwatch:PutMetricData",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:CreateLogGroup",
          "logs:DescribeLogStreams",
          "ecr:GetAuthorizationToken"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::<input-bucket>"
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject"
        ],
        "Resource": [
          "arn:aws:s3:::<input-bucket>/<object>",
          "arn:aws:s3:::<output-bucket>/<path>"
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repo>"
      }
    ]
  }
  ```
  Make note of the IAM role ARN. You will provide the role ARN created in this step when you configure the launch queue.
5. **Create an IAM role for the launch agent** The launch agent needs permission to create SageMaker training jobs. Attach the following policy to the IAM role that you will use for the launch agent. Make note of the IAM role ARN that you create for the launch agent:

  ```yaml
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "logs:DescribeLogStreams",
          "sagemaker:AddTags",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob"
        ],
        "Resource": "arn:aws:sagemaker:<region>:<account-id>:*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:PassRole",
        "Resource": "arn:aws:iam::<account-id>:role/<RoleArn-from-queue-config>"
      },
      {
        "Effect": "Allow",
        "Action": "kms:CreateGrant",
        "Resource": "<ARN-OF-KMS-KEY>",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "sagemaker.<region>.amazonaws.com",
            "kms:GrantIsForAWSResource": "true"
          }
        }
      }
    ]
  }
  ```
  
  


:::note
* If you want the launch agent to build images, see the [Advanced agent set up](./setup-agent-advanced.md) for additional permissions required.
* The `kms:CreateGrant` permission for SageMaker queues is required only if the associated ResourceConfig has a specified VolumeKmsKeyId and the associated role does not have a policy that permits this action.
:::

## Configure a queue for SageMaker
Create a queue in the W&B App that uses SageMaker as its compute resource:

1. Navigate to the [Launch App](https://wandb.ai/launch).
3. Click on the **Create Queue** button.
4. Select the **Entity** you would like to create the queue in.
5. Provide a name for your queue in the **Name** field.
6. Select **SageMaker** as the **Resource**.
7. Within the **Configuration** field, provide information about your SageMaker job. By default, W&B will populate a YAML and JSON `CreateTrainingJob` request body:
```json
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
You must at minimum specify:

- `RoleArn` : ARN of the IAM role that you created that satisfied the [prerequisites](#prerequisites).
- `OutputDataConfig.S3OutputPath` : An Amazon S3 URI specifying where SageMaker outputs will be stored.
- `ResourceConfig`: Required specification of a resource config. Options for resource config are outlined [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html).
- `StoppingCondition`: Required specification of the stopping conditions for the trainin job. Options outlined [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html).
7. Click on the **Create Queue** button.



## Configure a launch agent
Configure the launch agent with a YAML config file named `launch-config.yaml`. By default, W&B will check for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent.

The following YAML snippet demonstrates how to specify the core config agent options:

```yaml title="launch-config.yaml"
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

:::tip
There are two ways to submit launch on Amazon SageMaker:
* Option 1: Bring your own image (BYOI) and push it to your Amazon ECR repository.
* Option 2: Let the W&B Launch agent build a container for your and push it to your ECR repository.

You will need to provide additional information to your launch agent configuration if you want the launch agent to build images for you (Option 2). For more information, see [Advanced agent set up](./setup-agent-advanced.md).
:::


## Set up agent permissions for Amazon SageMaker
IAM roles can be associated with launch agents in a variety of ways. How you configure these roles will in part depend on where your launch agent is polling from.


Based on your use case, see the following guidance. 

### Agent polls from a local machine 

Use the AWS config files located at `~/.aws/config`  and `~/.aws/credentials` to associate a role with an agent that is polling on a local machine. Provide the IAM role ARN that you created for the launch agent in the previous step.
 
```yaml title="~/.aws/config"
[profile sagemaker-agent]
role_arn = arn:aws:iam::<account-id>:role/<agent-role-name>
source_profile = default                                                                   
```

```yaml title="~/.aws/credentials"
[default]
aws_access_key_id=<access-key-id>
aws_secret_access_key=<secret-access-key>
aws_session_token=<session-token>
```

<!-- Alternatively, you can use environment variables to specify your  -->

### Agent polls within AWS (such as EC2)
You can use an instance role to provide permissions to the agent if you want to run the agent within an AWS service like EC2. 
