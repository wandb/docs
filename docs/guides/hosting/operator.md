---
description: Deploy W&B Platform with Kubernetes Operator
displayed_sidebar: default
title: Run W&B Server on Kubernetes
---

## W&B Kubernetes Operator

Use the W&B Kubernetes Operator to simplify deploying, administering, troubleshooting, and scaling your W&B Server deployments on Kubernetes. You can think of the operator as a smart assistant for your W&B instance.

The W&B Server architecture and design continuously evolves to expand AI developer tooling capabilities, and to provide appropriate primitives for high performance, better scalability, and easier administration. That evolution applies to the compute services, relevant storage and the connectivity between them. To help facilitate continuous updates and improvements across deployment types, W&B users a Kubernetes operator.

:::info
W&B uses the operator to deploy and manage Dedicated Cloud instances on AWS, GCP and Azure public clouds.
:::

For more information about Kubernetes operators, see [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) in the Kubernetes documentation.

## Reasons for the architecture shift
Historically, the W&B application was deployed as a single deployment and pod within a Kubernetes Cluster or a single Docker container. W&B has, and continues to recommend, to externalize the Database and Object Store. Externalizing the Database and Object store decouples the application's state.

As the application grew, the need to evolve from a monolithic container to a distributed system (microservices) was apparent. This change facilitates backend logic handling and seamlessly introduces built-in Kubernetes infrastructure capabilities. Distributed systems also supports deploying new services essential for additional features that W&B relies on.

Before 2024, any Kubernetes-related change required manually updating the [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform module. Updating the Terraform module ensures compatibility across cloud providers, configuring necessary Terraform variables, and executing a Terraform apply for each backend or Kubernetes-level change. 

This process was not scalable since W&B Support had to assist each customer with upgrading their Terraform module.

The solution was to implement an operator that connects to a central [deploy.wandb.ai](https://deploy.wandb.ai) server to request the latest specification changes for a given release channel and apply them. Updates are received as long as the license is valid. [Helm](https://helm.sh/) is used as both the deployment mechanism for the W&B operator and the means for the operator to handle all configuration templating of the W&B Kubernetes stack, Helm-ception.

## How it works
You can install the operator with helm or from the source. See [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator) for detailed instructions. 

The installation process creates a deployment called `controller-manager` and uses a [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) definition named `weightsandbiases.apps.wandb.com` (shortName: `wandb`), that takes a single `spec` and applies it to the cluster:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

The `controller-manager` installs [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) based on the spec of the custom resource, release channel, and a user defined config. The configuration specification hierarchy enables maximum configuration flexibility at the user end and enables W&B to release new images, configurations, features, and Helm updates automatically.

For a detailed description of the specification hierarchy, see [Configuration Specification Hierarchy](#configuration-specification-hierarchy) and for configuration options, see [Configuration Reference](#configuration-reference-for-wb-operator).

## Configuration Specification Hierarchy
Configuration specifications follow a hierarchical model where higher-level specifications override lower-level ones. Here’s how it works:

- **Release Channel Values**: This base level configuration sets default values and configurations based on the release channel set by W&B for the deployment.
- **User Input Values**: Users can override the default settings provided by the Release Channel Spec through the System Console.
- **Custom Resource Values**: The highest level of specification, which comes from the user. Any values specified here override both the User Input and Release Channel specifications. For a detailed description of the configuration options, see [Configuration Reference](#configuration-reference-for-wb-operator).

This hierarchical model ensures that configurations are flexible and customizable to meet varying needs while maintaining a manageable and systematic approach to upgrades and changes.

## Requirements to use the W&B Kubernetes Operator
Satisfy the following requirements to deploy W&B with the W&B Kubernetes operator:

* Egress to the following endpoints during installation and during runtime:
    * deploy.wandb.ai
    * docker.io
    * quay.io
    * gcr.io
* A Kubernetes cluster at least version 1.28 with a deployed, configured and fully functioning Ingress controller (for example Contour, Nginx).
* Externally host and run MySQL 8.0 database.
* Object Storage (Amazon S3, Azure Cloud Storage, Google Cloud Storage, or any S3-compatible storage service) with CORS support.
* A valid W&B Server license.

See [this](./self-managed/bare-metal) guide for a detailed explanation on how to set up and configure a self-managed installation.

Depending on the installation method, you might need to meet the following requirements:
* Kubectl installed and configured with the correct Kubernetes cluster context.
* Helm is installed.

# Air-gapped installations
See the [Deploy W&B in airgapped environment with Kubernetes](./operator-airgapped) tutorial on how to install the W&B Kubernetes Operator in an airgapped environment.

# Deploy W&B Server application
This section describes different ways to deploy the W&B Kubernetes operator. 
:::note
The W&B Operator will become the default installation method for W&B Server. Other methods will be deprecated in the future.
:::

**Choose one of the following:**
- If you have provisioned all required external services and want to deploy W&B onto Kubernetes with Helm CLI, continue [here](#deploy-wb-with-helm-cli).
- If you prefer managing infrastructure and the W&B Server with Terraform, continue [here](#deploy-wb-with-helm-terraform-module).
- If you want to utilize the W&B Cloud Terraform Modules, continue [here](#deploy-wb-with-wb-cloud-terraform-modules).

## Deploy W&B with Helm CLI
W&B provides a Helm Chart to deploy the W&B Kubernetes operator to a Kubernetes cluster. This approach allows you to deploy W&B Server with Helm CLI or a continuous delivery tool like ArgoCD. Make sure that the above mentioned requirements are in place.

Follow those steps to install the W&B Kubernetes Operator with Helm CLI:

1. Add the W&B Helm repository. The W&B Helm chart is available in the W&B Helm repository. Add the repo with the following commands:
```shell
helm repo add wandb https://charts.wandb.ai
helm repo update
```
2. Install the Operator on a Kubernetes cluster. Copy and paste the following:
```shell
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```
3. Configure the W&B operator custom resource to trigger the W&B Server installation. Create an operator.yaml file to customize the W&B Operator deployment, specifying your custom configuration. See [Configuration Reference](#configuration-reference-for-wb-operator) for details.

Once you have the specification YAML created and filled with your values, run the following and the operator will apply the configuration and install the W&B Server application based on your configuration.

```shell
kubectl apply -f operator.yaml
```

Wait until the deployment is completed and verify the installation. This will take a few minutes.

4. Verify the installation. Access the new installation with the browser and create the first admin user account. When this is done, follow the verification steps as outlined [here](#verify-the-installation)


## Deploy W&B with Helm Terraform Module

This method allows for customized deployments tailored to specific requirements, leveraging Terraform's infrastructure-as-code approach for consistency and repeatability. The official W&B Helm-based Terraform Module is located [here](https://registry.terraform.io/modules/wandb/wandb/helm/latest). 

The following code can be used as a starting point and includes all necessary configuration options for a production grade deployment. 

```hcl
module "wandb" {
  source  = "wandb/wandb/helm"

  spec = {
    values = {
      global = {
        host    = "https://<HOST_URI>"
        license = "eyJhbGnUzaH...j9ZieKQ2x5GGfw"

        bucket = {
          <details depend on the provider>
        }

        mysql = {
          <redacted>
        }
      }

      ingress = {
        annotations = {
          "a" = "b"
          "x" = "y"
        }
      }

      # important, DO NOT REMOVE
      mysql = { install = false }
    }
  }
}
```

Note that the configuration options are the same as described in [Configuration Reference](#configuration-reference-for-wb-operator), but that the syntax has to follow the HashiCorp Configuration Language (HCL). The W&B custom resource definition will be created by the Terraform module.

To see how Weights&Biases themselves use the Helm Terraform module to deploy “Dedicated Cloud” installations for customers,  follow those links:
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

## Deploy W&B with W&B Cloud Terraform modules

W&B provides a set of Terraform Modules for AWS, GCP and Azure. Those modules deploy entire infrastructures including Kubernetes clusters, load balancers, MySQL databases and so on as well as the W&B Server application. The W&B Kubernetes Operator is already pre-baked with those official W&B cloud-specific Terraform Modules with the following versions:

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

This integration ensures that W&B Kubernetes Operator is ready to use for your instance with minimal setup, providing a streamlined path to deploying and managing W&B Server in your cloud environment.

For a detailed description on how to use these modules, refer to this [section](./hosting-options/self-managed#deploy-wb-server-within-self-managed-cloud-accounts) to self-managed installations section in the docs.

## Verify the installation

To verify the installation, W&B recommends using the [W&B CLI](../../ref/cli/README.md). The verify command executes several tests that verify all components and configurations. 

:::note
This step assumes that the first admin user account is created with the browser.
:::

Follow these steps to verify the installation:

1. Install the W&B CLI:
```shell
pip install wandb
```
2. Log in to W&B:
```shell
wandb login --host=https://YOUR_DNS_DOMAIN
```

For example:
```shell
wandb login --host=https://wandb.company-name.com
```

3. Verify the installation:
```shell
wandb verify
```

A successful installation and fully working W&B deployment shows the following output:

```console
Default host selected:  https://wandb.company-name.com
Find detailed logs for this test at: /var/folders/pn/b3g3gnc11_sbsykqkm3tx5rh0000gp/T/tmpdtdjbxua/wandb
Checking if logged in...................................................✅
Checking signed URL upload..............................................✅
Checking ability to send large payloads through proxy...................✅
Checking requests to base url...........................................✅
Checking requests made over signed URLs.................................✅
Checking CORs configuration of the bucket...............................✅
Checking wandb package version is up to date............................✅
Checking logged metrics, saving and downloading a file..................✅
Checking artifact save and download workflows...........................✅
``` 

## Access the W&B Management Console
The W&B Kubernetes operator comes with a management console. It is located at `${HOST_URI}/console`, for example `https://wandb.company-name.com/` console.

There are two ways to log in to the management console:

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="option1"
  values={[
    {label: 'Option 1 (Recommended)', value: 'option1'},
    {label: 'Option 2', value: 'option2'},
  ]}>
  <TabItem value="option1">

1. Open the W&B application in the browser and login. Log in to the W&B application with `${HOST_URI}/`, for example `https://wandb.company-name.com/`
2. Access the console. Click on the icon in the top right corner and then click on **System console**. Note that only users with admin privileges will see the **System console** entry.

![](/images/hosting/access_system_console_via_main_app.png)

  </TabItem>
  <TabItem value="option2">

:::note
W&B recommends you access the console using the following steps only if Option 1 does not work.
:::

1. Open console application in browser. Open the above described URL in the browser and you will be presented with this login screen:
![](/images/hosting/access_system_console_directly.png)
2. Retrieve password. The password is stored as a Kubernetes secret and is generated as part of the installation. To retrieve it, execute the following command:
```shell
kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
```
Copy the password to the clipboard.
3: Login to the console. Paste the copied password to the textfield “Enter password” and click login.


  </TabItem>
</Tabs>



## Update the W&B Kubernetes operator
This section describes how to update the W&B Kubernetes operator. 

:::note
* Updating the W&B Kubernetes operator does not update the W&B server application.
* See the instructions [here](#migrate-self-managed-instances-to-wb-operator) if you use a Helm chart that does not user the W&B Kubernetes operator before you follow the proceeding instructions to update the W&B operator.
:::

Copy and paste the code snippets below into your terminal. 

1. First, update the repo with [`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/):
```shell
helm repo update
```

2. Next, update the Helm chart with [`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/):
 
```shell
helm upgrade operator wandb/operator -n wandb-cr --reuse-values
```

## Update the W&B Server application
You no longer need to update W&B Server application if you use the W&B Kubernetes operator.

The operator automatically updates your W&B Server application when a new version of the software of W&B is released.


## Migrate self-managed instances to W&B Operator
The proceeding section describe how to migrate from self-managing your own W&B Server installation to using the W&B Operator to do this for you. The migration process depends on how you installed W&B Server:

:::note
The W&B Operator will become the default installation method for W&B Server. In the future, W&B will deprecate deployment mechanisms that do not use the operator. Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions.
:::

- If you used the official W&B Cloud Terraform Modules, navigate to the appropriate documentation and follow the steps there:
  - [AWS](#migrate-to-operator-based-aws-terraform-modules)
  - [GCP](#migrate-to-operator-based-gcp-terraform-modules)
  - [Azure](#migrate-to-operator-based-azure-terraform-modules)
- If you used the [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb),  continue [here](#migrate-to-operator-based-helm-chart).
- If you used the [W&B Non-Operator Helm chart with Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest),  continue [here](#migrate-to-operator-based-terraform-helm-chart).
- If you created the Kubernetes resources with manifest(s),  continue [here](#migrate-to-operator-based-helm-chart).


### Migrate to Operator-based AWS Terraform Modules

For a detailed description of the migration process,  continue [here](self-managed/aws-tf#migrate-to-operator-based-aws-terraform-modules).

### Migrate to Operator-based GCP Terraform Modules

Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions or need assistance.


### Migrate to Operator-based Azure Terraform Modules

Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions or need assistance.

### Migrate to Operator-based Helm chart

Follow these steps to migrate to the Operator-based Helm chart:

1. Get the current W&B configuration. If W&B was deployed with an non-operator-based version of the Helm chart,  export the values like this:
```shell
helm get values wandb
```
If W&B was deployed with Kubernetes manifests,  export the values like this:
```shell
kubectl get deployment wandb -o yaml
```
In both ways you should now have all the configuration values which are needed for the next step. 

2. Create a file called operator.yaml. Follow the format described in the [Configuration Reference](#configuration-reference-for-wb-operator). Use the values from step 1.

3. Scale the current deployment to 0 pods. This step is stops the current deployment.
```shell
kubectl scale --replicas=0 deployment wandb
```
4. Update the Helm chart repo:
```shell
helm repo update
```
5. Install the new Helm chart:
```shell
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```
6. Configure the new helm chart and trigger W&B application deployment. Apply the new configuration.
```shell
kubectl apply -f operator.yaml
```
The deployment will take a few minutes to complete.

7. Verify the installation. Make sure that everything works by following the steps in [Verify the installation](#verify-the-installation).

8. Remove to old installation. Uninstall the old helm chart or delete the resources that were created with manifests.

### Migrate to Operator-based Terraform Helm chart

Follow these steps to migrate to the Operator-based Helm chart:


1. Prepare Terraform config. Replace the Terraform code from the old deployment in your Terraform config with the one that is described [here](#deploy-wb-with-helm-terraform-module). Set the same variables as before. Do not change .tfvars file if you have one.
2. Execute Terraform run. Execute terraform init, plan and apply
3. Verify the installation. Make sure that everything works by following the steps in [Verify the installation](#verify-the-installation).
4. Remove to old installation. Uninstall the old helm chart or delete the resources that were created with manifests.



## Configuration Reference for W&B Server

This section describes the configuration options for W&B Server application. The application receives its configuration as custom resource definition named [WeightsAndBiases](#how-it-works). Some configuration options are exposed with the below configuration, some need to be set as environment variables.

The documentation has two lists of environment variables: [basic](./env-vars) and [advanced](./iam/advanced_env_vars). Only use environment variables if the configuration option that you need are not exposed using Helm Chart.

The W&B Server application configuration file for a production deployment requires the following contents:

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/name: weightsandbiases
    app.kubernetes.io/instance: wandb
  name: wandb
  namespace: default
spec:
  values:
    global:
      host: https://<HOST_URI>
      license: eyJhbGnUzaH...j9ZieKQ2x5GGfw
      bucket:
        <details depend on the provider>
      mysql:
        <redacted>
    ingress:
      annotations:
        <redacted>
    mysql:
      # important, DO NOT REMOVE
      install: false
```

This YAML file defines the desired state of your W&B deployment, including the version, environment variables, external resources like databases, and other
necessary settings. Use the above YAML as a starting point and add the missing information.

The full list of spec customization can be found [here](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml) in the Helm repository. The recommended approach is to only change what is necessary and otherwise use the default values.


### Complete example 
This is an example configuration that uses GCP Kubernetes with GCP Ingress and GCS (GCP Object storage):

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/name: weightsandbiases
    app.kubernetes.io/instance: wandb
  name: wandb
  namespace: default
spec:
  values:
    global:
      host: https://abc-wandb.sandbox-gcp.wandb.ml
      bucket:
        name: abc-wandb-moving-pipefish
        provider: gcs
      mysql:
        database: wandb_local
        host: 10.218.0.2
        name: wandb_local
        password: 8wtX6cJHizAZvYScjDzZcUarK4zZGjpV
        port: 3306
        user: wandb
      license: eyJhbGnUzaHgyQjQyQWhEU3...ZieKQ2x5GGfw
    ingress:
      annotations:
        ingress.gcp.kubernetes.io/pre-shared-cert: abc-wandb-cert-creative-puma
        kubernetes.io/ingress.class: gce
        kubernetes.io/ingress.global-static-ip-name: abc-wandb-operator-address
    mysql:
      install: false
```


### Host
```yaml
 # Provide the FQDN with protocol
global:
  # example host name,  replace with your own
  host: https://abc-wandb.sandbox-gcp.wandb.ml
```

### Object storage (bucket)

**AWS**
```yaml
global:
  bucket:
    provider: "s3"
    name: ""
    kmsKey: ""
    region: ""
```

**GCP**
```yaml
global:
  bucket:
    provider: "gcs"
    name: ""
```

**Azure**
```yaml
global:
  bucket:
    provider: "az"
    name: ""
    secretKey: ""
```

**Other providers (Minio, Ceph, etc.)**

For other S3 compatible providers, set the bucket configuration as a environment variable as follows:
```yaml
global:
  extraEnv:
    "BUCKET": "s3://wandb:changeme@mydb.com/wandb?tls=true"
```
The variable contains a connection string in this form:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

You can optionally tell W&B to only connect over TLS if you configure a trusted SSL certificate for your object store. To do so, add the `tls` query parameter to the url:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```
:::caution
This will only work if the SSL certificate is trusted. W&B does not support self-signed certificates.
:::

### MySQL

```yaml
global:
   mysql:
     # Example values, replace with your own
     database: wandb_local
     host: 10.218.0.2
     name: wandb_local
     password: 8wtX6cJH...ZcUarK4zZGjpV
     port: 3306
     user: wandb
```

### License

```yaml
global:
  # Example license,  replace with your own
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

### Ingress

To identify the ingress class,  see this FAQ [entry](#how-to-identify-the-kubernetes-ingress-class).

**Without TLS**

```yaml
global:
# IMPORTANT: Ingress is on the same level in the YAML as ‘global’ (not a child)
ingress:
  class: ""
```

**With TLS**

Create a secret that contains the certificate

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

Reference the secret in the ingress configuration
```yaml
global:
# IMPORTANT: Ingress is on the same level in the YAML as ‘global’ (not a child)
ingress:
  class: ""
  annotations:
    {}
    # kubernetes.io/ingress.class: nginx
    # kubernetes.io/tls-acme: "true"
  tls: 
    - secretName: wandb-ingress-tls
      hosts:
        - <HOST_URI>
```

In case of Nginx you might have to add the following annotation:

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### Custom Kubernetes ServiceAccounts

Specify custom Kubernetes service accounts to run the W&B pods. 

The following snippet creates a service account as part of the deployment with the specified name:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: true

parquet:
  serviceAccount:
    name: custom-service-account
    create: true

global:
  ...
```
The subsystems "app" and "parquet" runs under the specified service account. The other subsystem runs under the default service account.

If the service account already exists on the cluster, set `create: false`:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: false

parquet:
  serviceAccount:
    name: custom-service-account
    create: false
    
global:
  ...
```



You cana specify service accounts on different subsystems such as app, parquet, console, and more:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: true

console:
  serviceAccount:
    name: custom-service-account
    create: true

global:
  ...
```

The service accounts can be different between the subsystems:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: false

console:
  serviceAccount:
    name: another-custom-service-account
    create: true

global:
  ...
```

### External Redis

```yaml
redis:
  install: false

global:
  redis:
    host: ""
    port: 6379
    password: ""
    parameters: {}
    caCert: ""
```

Alternatively with redis password in a Kubernetes secret:

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

Reference it in below configuration:
```yaml
redis:
  install: false

global:
  redis:
    host: redis.example
    port: 9001
    auth:
      enabled: true
      secret: redis-secret
      key: redis-password
```

### LDAP
**Without TLS**
```yaml
global:
  ldap:
    enabled: true
    # LDAP server address including "ldap://" or "ldaps://"
    host:
    # LDAP search base to use for finding users
    baseDN:
    # LDAP user to bind with (if not using anonymous bind)
    bindDN:
    # Secret name and key with LDAP password to bind with (if not using anonymous bind)
    bindPW:
    # LDAP attribute for email and group ID attribute names as comma separated string values.
    attributes:
    # LDAP group allow list
    groupAllowList:
    # Enable LDAP TLS
    tls: false
```

**With TLS**

The LDAP TLS cert configuration requires a config map pre-created with the certificate content.

To create the config map you can use the following command:

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

And use the config map in the YAML like the example below

```yaml
global:
  ldap:
    enabled: true
    # LDAP server address including "ldap://" or "ldaps://"
    host:
    # LDAP search base to use for finding users
    baseDN:
    # LDAP user to bind with (if not using anonymous bind)
    bindDN:
    # Secret name and key with LDAP password to bind with (if not using anonymous bind)
    bindPW:
    # LDAP attribute for email and group ID attribute names as comma separated string values.
    attributes:
    # LDAP group allow list
    groupAllowList:
    # Enable LDAP TLS
    tls: true
    # ConfigMap name and key with CA certificate for LDAP server
    tlsCert:
      configMap:
        name: "ldap-tls-cert"
        key: "certificate.crt"
```

### OIDC SSO

```yaml
global: 
  auth:
    sessionLengthHours: 720
    oidc:
      clientId: ""
      secret: ""
      authMethod: ""
      issuer: ""
```

### SMTP

```yaml
global:
  email:
    smtp:
      host: ""
      port: 587
      user: ""
      password: ""
```

### Environment Variables
```yaml
global:
  extraEnv:
    GLOBAL_ENV: "example"
```

### Custom certificate authority
`customCACerts` is a list and can take many certificates. Certificate authorities specified in `customCACerts` only apply to the W&B Server application.

```yaml
global:
  customCACerts:
  - |
    -----BEGIN CERTIFICATE-----
    MIIBnDCCAUKgAwIBAg.....................fucMwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9tZU.....................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMFoXDT.....................oNWYggsMo8O+0mWLYMAoGCCqG
    SM49BAMCA0gAMEUCIQ.....................hwuJgyQRaqMI149div72V2QIg
    P5GD+5I+02yEp58Cwxd5Bj2CvyQwTjTO4hiVl1Xd0M0=
    -----END CERTIFICATE-----
  - |
    -----BEGIN CERTIFICATE-----
    MIIBxTCCAWugAwIB.....................qaJcwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
    SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
    aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
    -----END CERTIFICATE-----
```

## Configuration Reference for W&B Operator

This section describes configuration options for W&B Kubernetes operator (`wandb-controller-manager`). The operator receives its configuration in the form of a YAML file. 

By default, the W&B Kubernetes operator does not need a configuration file. Create a configuration file if required. For example, you might need a configuration file to specify custom certificate authorities, deploy in an air gap environment and so forth. 

Find the full list of spec customization [in the Helm repository](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

### Custom CA
A custom certificate authority (`customCACerts`), is a list and can take many certificates. Those certificate authorities when added only apply to the W&B Kubernetes operator (`wandb-controller-manager`). 

```yaml
customCACerts:
- |
  -----BEGIN CERTIFICATE-----
  MIIBnDCCAUKgAwIBAg.....................fucMwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9tZU.....................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMFoXDT.....................oNWYggsMo8O+0mWLYMAoGCCqG
  SM49BAMCA0gAMEUCIQ.....................hwuJgyQRaqMI149div72V2QIg
  P5GD+5I+02yEp58Cwxd5Bj2CvyQwTjTO4hiVl1Xd0M0=
  -----END CERTIFICATE-----
- |
  -----BEGIN CERTIFICATE-----
  MIIBxTCCAWugAwIB.....................qaJcwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
  SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
  aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
  -----END CERTIFICATE-----
```

## FAQ

#### How to get the  W&B Operator Console password
See [Accessing the W&B Kubernetes Operator Management Console](#access-the-wb-management-console).


#### How to access the W&B Operator Console if Ingress doesn’t work

Execute the following command on a host that can reach the Kubernetes cluster:

```console
kubectl port-forward svc/wandb-console 8082
```

Access the console in the browser with `https://localhost:8082/` console.

See [Accessing the W&B Kubernetes Operator Management Console](#access-the-wb-management-console) on how to get the password (Option 2).

#### How to view W&B Server logs

The application pod is named **wandb-app-xxx**.

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

#### How to identify the Kubernetes ingress class

You can get the ingress class installed in your cluster by running

```console
kubectl get ingressclass
```
