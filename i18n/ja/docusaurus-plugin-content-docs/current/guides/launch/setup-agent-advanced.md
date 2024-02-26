---
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Advanced agent set up
How you configure the launch agent will depend on numerous factors. One of those factors is whether or not the launch agent will build an image for you. 

:::tip
The W&B launch agent will build an image for you if you provide a Git repo based or artifact based jobs.
:::

In the simplest use case, you provide an image-based launch job that is executed in a launch queue target environment that has access to your image repository More requirements must be satisfied if you use the launch agent to build images for you. 


## Builders
Launch agents build images for W&B artifacts and Git repo sourced jobs. This means that the launch agent config file (`launch-config.yaml`) must have a builder option specified. W&B Launch supports two builders: Kaniko and Docker. 

We suggest that you use either Kaniko or Docker based on the following scenarios:

* Kaniko: Use Kaniko when the agent polls launch queues in a Kubernetes cluster
* Docker: Use Docker for all other cases.

### Docker
We recommend that you use the Docker builder if you want the agent to build images on a local machine (that has Docker installed). Specify the Docker builder in the launch agent config with the builder key. 

For example, the following YAML snippet shows how to specify this in a launch agent config file (`launch-config.yaml`):

```yaml title="launch-config.yaml"
builder:
  type: docker
```

### Kaniko
To use the Kaniko builder, you must specify a container registry and environment option.

For example, the following YAML snippet shows how to specify Kaniko in a launch agent config file (`launch-config.yaml`):

```yaml title="launch-config.yaml"
builder:
  type: kaniko
  build-context-store: s3://my-bucket/build-contexts/ 
  build-job-name: wandb-image-build # Kubernetes job name prefix for all builds
```
<!-- For specific policies the Kaniko job can use to interact with the context store see Put in Bucket[LINK]. -->

If you run a Kubernetes cluster other than using AKS, EKS, or GKE, you will need to create a Kubernetes secret that contains the credentials for your cloud environment.

- To grant access to GCP, this secret should contain a [service account json](https://cloud.google.com/iam/docs/keys-create-delete#creating).
- To grant access to AWS, this secret should contain an [AWS credentials file](https://docs.aws.amazon.com/sdk-for-php/v3/developer-guide/guide_credentials_profiles.html).

Within your agent configuration file, and within the builder section, set the `secret-name` and `secret-key` keys to let Kaniko use the secrets:

```yaml title="launch-config.yaml"
builder:
	type: kaniko
  build-context-store: <my-build-context-store>
  secret-name: <Kubernetes-secret-name>
  secret-key: <secret-file-name>
```

:::note
The Kaniko builder requires permissions to put data into cloud storage (such as Amazon S3) see the [Agent permissions](#agent-permissions) section for more information.
:::

## Connect an agent to a cloud registry
You can connect the launch agent to a cloud container registry such Amazon Elastic Container Registry (Amazon ECR), Google Artifact Registry on GCP, or Azure Container Registry. The following describes common use cases as to why you might want to connect the launch agent to a cloud container registry:

- you do not want to store images you are building on your local machine
- you want to share images across multiple machines
- if the agent builds an image for you and you use a cloud compute resource such as Amazon SageMaker or VertexAI.


To connect the launch agent to a cloud container registry, you will need to provide additional information about the cloud environment and registry you want to use in the launch agent config. In addition, you will need to grant the agent permissions within the cloud environment to interact with required components based on your use case.

### Agent configuration
Within your launch agent config (`launch-config.yaml`), provide the name of the target resource environment and the container registry for the `environment` and `registry` keys, respectively.

The following tabs demonstrates how to configure the launch agent based on your environment and registry.


<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'GCP', value: 'gcp'},
    {label: 'Azure', value: 'azure'},
  ]}>
  <TabItem value="aws">

The AWS environment configuration requires the region key to be set. The region should be the AWS region that the agent will be running in. When the agent starts, it will use boto3 to load the default AWS credentials. 

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
registry:
  type: ecr
  # URI of the ECR repository where the agent will store images.
  # Make sure the region matches what you have configured in your
  # environment.
  uri: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Alternatively, you can simply set the repository name
  # repository: my-repository-name
```
See the [boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for more information on how to configure default AWS credentials.

  </TabItem>
  <TabItem value="gcp">

The GCP environment requires the region and project keys to be set. The region should be the GCP region that the agent will be running in. The project should be the GCP project that the agent will be running in. When the agent starts, it will use `google.auth.default()` to load the default GCP credentials. 

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
registry:
  # Requires a gcp environment configuration.
  type: gcr
  # URI of the Artifact Registry repository and image name where the agent
  # will store images. Make sure the region and project match what you have
  # configured in your environment.
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Alternatively, you may set the repository and image-name keys.
  # repository: my-artifact-repo
  # image-name: my-image-name
```

See the [`google-auth` documentation](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default for more information on how to configure default GCP credentials.

  </TabItem>
  <TabItem value="azure">

The Azure environment does not require any additional keys to be set. When the agent starts, it will use `azure.identity.DefaultAzureCredential()` to load the default Azure credentials. 

```yaml title="launch-config.yaml"
environment:
  type: azure
registry:
  type: acr
  uri: https://my-registry.azurecr.io/my-repository
```

See the [`azure-identity` documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) for more information on how to configure default Azure credentials.

  </TabItem>
</Tabs>


## Agent permissions
The agent permissions required will depend on your use case. The policies outlined below are used by launch agents.
### Cloud registry permissions
Below are the permissions that are generally required by launch agents to interact with cloud registries.

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'GCP', value: 'gcp'},
    {label: 'Azure', value: 'azure'},
  ]}>
  <TabItem value="aws">

```yaml
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:CreateRepository",
        "ecr:UploadLayerPart",
        "ecr:PutImage",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:DescribeImages",
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchDeleteImage"
      ],
      "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repository>"
    },
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    }
  ]
}
```

  </TabItem>
  <TabItem value="gcp">

```js
artifactregistry.dockerimages.list
artifactregistry.repositories.downloadArtifacts
artifactregistry.repositories.list
artifactregistry.repositories.uploadArtifacts
```

  </TabItem>
  <TabItem value="azure">

Add the [`AcrPush` role](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush) if you use the Kaniko builder.

</TabItem>
</Tabs>

### Kaniko permissions
The launch agent requires permission to push to cloud storage if the agent uses the Kaniko builder. Kaniko uses a context store outside of the pod running the build job.


<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'GCP', value: 'gcp'},
    {label: 'Azure', value: 'azure'},
  ]}>
  <TabItem value="aws">

The recommended context store for the Kaniko builder on AWS is Amazon S3. The following policy can be used to give the agent access to an S3 bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
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

  </TabItem>
  <TabItem value="gcp">

On GCP, the following IAM permissions are required for the agent to upload build contexts to GCS:

```js
storage.buckets.get
storage.objects.create
storage.objects.delete
storage.objects.get
```

  </TabItem>
  <TabItem value="azure">

The [Storage Blob Data Contributor](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) role is required in order for the agent to upload build contexts to Azure Blob Storage.


  </TabItem>
</Tabs>



### Permissions to execute jobs
The agent needs permission in your AWS or GCP cloud to start jobs on Sagemaker or Vertex AI, respectively.

<Tabs
  defaultValue="aws"
  values={[
    {label: 'Amazon SageMaker', value: 'aws'},
    {label: 'Vertex AI', value: 'vertex'},
  ]}>
  <TabItem value="aws">

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
```

:::note
The `kms:CreateGrant` permission for SageMaker queues is required only if the associated ResourceConfig has a specified VolumeKmsKeyId and the associated role does not have a policy that permits this action.
:::

  </TabItem>
  <TabItem value="vertex">

In order to run jobs with vertex AI you will also need to set up a GCS bucket and grant the agent the permissions described above.

```js
ml.jobs.create
ml.jobs.list
ml.jobs.get
```

  </TabItem>
</Tabs>