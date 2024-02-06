---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for SageMaker

<!-- You can use W&B Launch to submit jobs to run as SageMaker training jobs. Amazon SageMaker training jobs allow users to train machine learning models using provided or custom algorithms on the SageMaker platform. Once initiated, SageMaker handles the underlying infrastructure, scaling, and orchestration. -->

You can use W&B Launch to submit launch jobs to Amazon SageMaker to train machine learning models using provided or custom algorithms on the SageMaker platform. SageMaker takes care of spinning up and releasing compute resources, so it can be a good choice for teams without an EKS cluster.

Launch jobs sent to a Launch queue connected to Sagemaker are executed as SageMaker Training Jobs with the [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html). Arguments to the `CreateTrainingJob` API are controlled with the launch queue configuration.  

Amazon SageMaker [uses Docker images to execute training jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html). Images pulled by SageMaker must be stored in the Amazon Elastic Container Registry (ECR). This means that the image you use for training must be stored on ECR. 

:::note
This guide is for executing Sagemaker Training Jobs.  For deploying to  Sagemaker inference, see [this example Launch job](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints).
:::


## Prerequisites

### Decide whether you want the Launch agent to build images for you

 As with other queue types, for Sagemaker Launch agents can either:

 * (A) Run existing images with overrides specified through W&B, or
 * (B) Build an image for you, add it to ECR, and submit it to the Sagemaker training service along with the overrides. 

(A) is faster per-job and works well with existing CI systems, while (B), once set up, can offer some simplicity to ML Engineers rapidly iterating over training code.  

For (B), you will need to provide additional information to your launch agent configuration allowing it to build. For more information, see [Advanced agent set up](./setup-agent-advanced.md).


### Gather information about S3, ECR, and Sagemaker IAM roles

If you actively use Sagemaker in AWS already, you likely already have:

1. An [ECR repository](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) to store container images.
2. One or more [S3 buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) to store inputs and outputs for your Sagemaker training jobs.
3. An IAM role for Sagemaker to run training jobs and interact with ECR and S3 (see the [[walkthough]] for an example).

Make a note of the ARNs for these resources, because you'll use them in the Launch queue configuration.  If you don't have these resources, create them in AWS or follow our walkthrough tutorial [[link]].

### Create an IAM role for Launch agent

The Launch agent needs permission to create SageMaker training jobs

1. From the IAM screen in AWS, create a new role. 
2. For Trusted Entity, select `AWS Account` (or another option suiting your organization's policies).
3. Scroll through the permissions screen and click Next--you will create a new permissions policy elsewhere.
4. Give the role a name and description (e.g. `launch-agent-role`).
5. Create the role, then view it. Under `Add permissions`, select `Create inline policy`.
6. Toggle to the JSON policy editor, then paste the following, substituting your values:


  ```json
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

Click Next and make note of the ARN for the role--this will be used in setting up the agent. 

:::note
* If you want the launch agent to build images, see the [Advanced agent set up](./setup-agent-advanced.md) for additional permissions required.
* The `kms:CreateGrant` permission for SageMaker queues is required only if the associated ResourceConfig has a specified VolumeKmsKeyId and the associated role does not have a policy that permits this action.
:::

## Configure a queue for SageMaker

Next, create a queue in the W&B App that uses SageMaker as its compute resource:

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

- `RoleArn` : ARN of the Sagemaker execution IAM role (see [prerequisites](#prerequisites)). Not to be confused with the launch **agent** IAM role.
- `OutputDataConfig.S3OutputPath` : An Amazon S3 URI specifying where SageMaker outputs will be stored.
- `ResourceConfig`: Required specification of a resource config. Options for resource config are outlined [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ResourceConfig.html).
- `StoppingCondition`: Required specification of the stopping conditions for the training job. Options outlined [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html).
7. Click on the **Create Queue** button.


## Start the launch agent

### Decide where to run the Launch agent

There are several options for how the Launch agent is deployed for a Sagemaker queue: on a local machine, on an EC2 instance, or in an EKS cluster.

For production workloads and for customers who already have an EKS cluster, W&B recommends deploying the Launch agent to the EKS cluster using this Helm chart.

For production workloads without an current EKS cluster, an EC2 instance is a good approach--while the agent instance will keep running all the time, the agent doesn't need more than a `t2.micro`-sized instance which is relatively affordable.

For experimental or solo use cases, running the Launch agent on your local machine can be a fast way to get started.


#### Agent running in EKS

[This helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) can be used to install the agent in an EKS cluster.

#### Agent running in EC2

1. Go to `EC2` in the AWS console and click `Launch instance`.
2. Pick a t2.micro and use an appropriate security group and key pair for your organization.
3. Expand `Advanced details` and for `IAM instance profile`, select the launch agent IAM role you created above.
4. Launch the instance.

Once the instance is running, connect to it and `pip install wandb[launch]`.  **Need to Make sure python is installed.  Also how do you associate with the right role**


#### Agent running from a local machine 

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

Note that session tokens have a [max length](https://docs.aws.amazon.com/cli/latest/reference/sts/get-session-token.html#description) of 1 hour or 3 days depending on the principal they are associated with.

###  Configure a launch agent
Configure the launch agent with a YAML config file named `launch-config.yaml`. 

By default, W&B will check for the config file in `~/.config/wandb/launch-config.yaml`. You can optionally specify a different directory when you activate the launch agent with the `-c` flag.

The following YAML snippet demonstrates how to specify the core config agent options:

```yaml title="launch-config.yaml"
max_jobs: -1
queues:
  - <queue-name>
environment:
  type: aws
  region: <your-region>
registry:
  type: ecr
  uri: <ecr-repo-arn>
builder: 
  type: docker

```


## Push the job images in ECR

If you are going to have Launch run existing images with overrides specified through W&B, rather than having the Launch agent build the image for you, you need to have the images already in the ECR repo.  The full URI to the image can then be used in job creation e.g.

```
wandb job create image <your-image-uri> --p <project> ...
```


<!-- Alternatively, you can use environment variables to specify your  -->
