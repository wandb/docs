# Launch on SageMaker

Use W&B Launch to send your runs to AWS SageMaker.

## AWS Setup

You will need to create a few AWS resources in order to launch your runs onto AWS SageMaker. The steps below are necessary to use SageMaker in general, so if you are already using SageMaker, it's likely that the resources described below already exist in your AWS account.

### Prerequisites

To use W&B Launch with AWS SageMaker, you will need to setup SageMaker in your AWS account. For instructions on how to do so, follow [this guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-set-up.html) from the official SageMaker docs.

### Execution role

[Create an IAM role with AmazonSageMakerFullAccess policy for your SageMaker jobs to assume](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

### Container registry

[Create an ECR repository](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) to store any container images that you want to execute on SageMaker. You can launch custom container images that you have pushed to that repository yourself or configure the launch agent to build container images and push them to your ECR repository.

### Storage

[Create an S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) that can be used to store SageMaker outputs from your runs.

## Agent setup

In order for the launch agent to launch jobs with SageMaker you will need configure AWS credentials for you agent. Set the AWS credentials you want the agent to use via environment variables or as the `default` profile in your AWS config and then add an `environment` block to your agent config.

```yaml
environment:
	type: aws
	region: <aws-region>  # E.g. us-east-2
```

Any containers you want to run in SageMaker will need to be stored in your Elastic Container Registry (ECR). If you want your agent to build new customers and push them to ECR for you, you will need to add a `registry` block to your agent config.

```yaml
registry:
	type: ecr
	repository: <ecr-repo-name>
```

The agent can build container images with either Kaniko or Docker. The agent will attempt to perform container builds with `docker build` by default. If you are running the agent in Kubernetes you can enable Kaniko builds by adding the following to you agent config:

```yaml
builder:
	type: kaniko
	build-context-store: s3://<s3-bucket>/<prefix>
```

Kaniko will store compressed build contexts in the local specified under `build-context-store` and then push any container images it builds to the ECR repository configured in the `registry` block. Kaniko pods will need permission 

## Queue setup

The config for a SageMaker queue is a JSON blob that gets passed to [SageMaker API's `CreateTrainingJob` request](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html). When you create the queue, the config will be automatically populated with the following JSON:

```yaml
{
    "RoleArn": "<REQUIRED>",
    "EcrRepoName": "",
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

You can arguments you like, but you must specify:

- `RoleArn` : ARN of the role the IAM role that will be assumed by the job.
- `OutputDataConfig.S3OutputPath` : An S3 URI specifying where SageMaker outputs will be stored.
