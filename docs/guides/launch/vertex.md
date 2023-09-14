---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up for Vertex

Use W&B Launch to send your runs to GCP Vertex. There are two ways to use Launch on Vertex:

1. Bring your own image (BYOI) and push it to your GCP Artifact Registry.
2. Let the W&B Launch agent build a container for your and push it to your Artifact Registry.

The following table highlights the key differences between the two workflows listed above:

|                             | BYOI                    | Default W&B Launch                                       |
| --------------------------- | ----------------------- | -------------------------------------------------------- |
| Allowed job type            | Image source-job        | Git or code artifact sourced job                         |
| Queue configuration options | Same for both workflows | Same for both workflows                                  |
| Agent configuration options | N/A                     | Must have the `registry` block in your agent config file |
| Builder options             | Docker, Kaniko, Noop    | Docker, Kaniko                                           |

:::tip
**Should I bring my own image or let W&B build and push the image for me?**

Follow the bring your own image (BYOI) method if you uses packages that are not available on PyPi or find that you are having issues with W&B building your images.
In which case, we appreciate feedback about the specific problem so we can fix it in future releases.
:::

The following sections outline the steps to set up and run a job on Vertex with Launch.

## Prerequisites

Create and make note of the following GCP resources:

1. **Setup GCP Vertex in your GCP account.** See the [GCP Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/start) for more information.

## Setup Google Cloud authentication

W&B Launch uses the Google Cloud Python SDK to interact with GCP systems like GCS and Artifact Registry. Follow these steps to set up your GCP authentication for the python SDK:

1. **Install the necessary Google Cloud Python SDKs** by running `pip install --upgrade 'wandb[launch]'`.
2. **Create a service account** with the necessary permissions to access GCP systems like GCS and Artifact Registry. See the [GCP IAM documentation](https://cloud.google.com/iam/docs/creating-managing-service-accounts) for more information.
3. **Download the JSON key** file for the service account.
4. **Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable** to the path of the JSON key file.
5. **Create a GCP Artifact Registry repository** to store images you want to execute on Vertex. See the [GCP Artifact Registry documentation](https://cloud.google.com/artifact-registry/docs/overview) for more information.
6. **(If W&B creates your image) Create a GCS bucket** to store build contexts. See the [GCP Storage documentation](https://cloud.google.com/storage/docs/creating-buckets) for more information.
7. **Create a staging GCS bucket** for Vertex AI to store its metadata. Note that this bucket must be in the same region as your Vertex AI workloads in order to be used as a staging bucket.
8. **Grant your service account permission** to access the GCS bucket and Artifact Registry repository. See the [GCP IAM documentation](https://cloud.google.com/iam/docs/creating-managing-service-accounts) for more information.

## Create a queue

Create a queue in the W&B App that uses Vertex as its compute resource:

1. Navigate to the [Launch page](https://wandb.ai/launch).
2. Click on the **Create Queue** button.
3. Select the **Entity** you would like to create the queue in.
4. Provide a name for your queue in the **Name** field.
5. Select **Vertex** as the **Resource**.
6. Within the **Configuration** field, provide configuration for how runs should be launched.

<!-- TODO: Decide on and put in default configs + docs on what they mean -->

<Tabs
defaultValue="JSON"
values={[
{label: 'JSON', value: 'JSON'},
{label: 'YAML', value: 'YAML'},
]}>
<TabItem value="JSON">

```json
{
  "spec": {
    "worker_pool_specs": [
      {
        "machine_spec": {
          "machine_type": "n1-standard-4",
          "accelerator_type": "ACCELERATOR_TYPE_UNSPECIFIED",
          "accelerator_count": 0
        },
        "replica_count": 1,
        "container_spec": {
          "image_uri": "${image_uri}"
        }
      }
    ],
    "staging_bucket": "<REQUIRED>"
  },
  "run": {
    "restart_job_on_worker_restart": false
  }
}
```

  </TabItem>
  <TabItem value="YAML">

```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
        accelerator_type: ACCELERATOR_TYPE_UNSPECIFIED
        accelerator_count: 0
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: <REQUIRED>
run:
  restart_job_on_worker_restart: false
```

  </TabItem>
</Tabs>

7. After you configure your queue, click on the **Create Queue** button.

## Vertex queue configuration

The Vertex AI resource arguments are stored under the `spec` and `run` keys.
The `spec` key contains values for the named arguments of the [`CustomJob` constructor](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec) in the Vertex AI Python SDK. The `run` key contains values for the named arguments of the `run` method of the `CustomJob` class in the Vertex AI Python SDK.
in the Vertex AI Python SDK. The `run` key contains values for the named arguments of
the `run` method of the `CustomJob` class in the Vertex AI Python SDK. For more
information of how to customize your Vertex jobs, see the [Vertex AI documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomJob#google_cloud_aiplatform_CustomJob)

Most useful customization should happen in the `spec.worker_pool_specs` list.
A worker pool spec defines a group of workers that will run your job. The worker
spec in the default config asks for a single `n1-standard-4` machine with no
accelerators. You can change the machine type and accelerator type and count
to suit your needs. For more information on the available machine types and
accelerator types, see the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec).

:::caution
Some of the VertexAI docs show worker pool specifications with all keys in
camel case, e.g. `workerPoolSpecs`. The Vertex AI Python SDK uses snake case
for these keys, e.g. `worker_pool_specs`. The launch queue configuration
should use snake case.
:::

## Configure the launch agent

Configure a launch agent to execute jobs from your queues with Vertex. The following steps outline how to configure your launch agent to use Vertex with Launch.

1. **Set the GCP credentials** you want the agent to use either by:

   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your GCP service account key file.
   - Use the `gcloud auth application-default login` command to set the default credentials for your local machine.
   - Set up a [GCP workload identity pool](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) and configure your agent to use it.

   Google's APIs support many additional authentication methods, the agent will work with any
   form of authentication that can loaded by [`google.auth.default()`](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default)

2. **Define the agent config**. Add the `environment` block in your agent config file (`~/.config/wandb/launch-config.yaml`).

   The following code snippet shows an example `launch-config.yaml` file. Ensure you specify the type as AWS and the region of your GCS bucket and Artifact Registry:

   ```yaml title="~/.config/wandb/launch-config.yaml"
   environment:
     type: gcp
     region: <gcp-region> # E.g. us-central1
     project: <gcp-project-name>
   ```

:::note
Continue to complete the following steps if you want W&B to build and push your image for you.

Skip the next two steps and move on to the [Add jobs to your queue section] if you brought your own imaged and you pushed that image to your Artifact Registry.
:::

3. **(Optional) Specify a `registry`**: If you want the W&B agent to build new containers and push them to ECR for you, you will need to add a `registry` block to your agent config.

   ```yaml title="~/.config/wandb/launch-config.yaml"
   registry:
     type: gcr
     uri: <artifact-registry-uri>, # E.g. us-central1-docker.pkg.dev/<gcp-project-name>/<artifact-registry-name>/<image-name>
   ```

4. **(Optional) Enable Kaniko**
   If you run the agent in Kubernetes you can enable Kaniko builds by adding the following to you agent config:

   ```yaml title="~/.config/wandb/launch-config.yaml"
   builder:
     type: kaniko
     build-context-store: gs://<gcs-bucket>/<prefix>
   ```

   Kaniko will store compressed build contexts in the local specified under `build-context-store` and then push any container images it builds to the Artifact Registry configured in the `registry` block. Kaniko pods will need permission to access the GCS bucket specified in `build-context-store` and read/write access to the Registy repository specified in `registry.uri`.

## Add jobs to your queue

Follow these steps to add your launch job to your queue:

1. Navigate to your W&B project where the job is defined.
2. Select the **Jobs** icon on the left panel (thunderbolt image). This will redirect you to the Jobs page within your project workspace.
3. Hover your mouse on the right side of the job name and click on the **Launch** button that appears. A drawer will appear from the right side of your screen. Provide the following:
   - The name your queue from the **Queue** dropdown menu. If you have not created a queue yet, see the [Create a queue section](#1-create-a-queue).
   - Select a W&B project from the **Destination project** dropdown menu.
4. Click **Launch now**.

## Start your agent

Run the agent locally, in a Kubernetes cluster, or in a Docker container. The launch agent will continuously run launch jobs on Vertex so long as the agent is an environment with GCP credentials.

Copy and paste the following. Ensure to replace values in `<>` with your own values:

```bash
wandb launch-agent -e <your-entity> -q <queue-name>  \\
    -c <path-to-agent-config>
```

Another common pattern is to run the agent on a GCP Compute Engine instance. The agent can perform container builds and push them to Artifact Registry if you install Docker on the Compute Engine instance where the agent is running. The launch agent can then launch jobs on Vertex AI using the GCP credentials associated with the Compute Engine instance. Google provides a guide to installing Docker on Compute Engine instances [here](https://cloud.google.com/compute/docs/containers/deploying-containers#installing_docker_on_a_compute_engine_instance).
