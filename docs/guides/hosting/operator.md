---
description: Hosting W&B Server via Kubernetes Operator
displayed_sidebar: default
---

# W&B Kubernetes Operator

Use the W&B Kubernetes Operator to simplify deploying, administering, troubleshooting, and scaling your W&B Server deployments on Kubernetes. You can think of the operator as a smart assistant for your W&B instance.

The W&B Server architecture and design continuously evolve to expand the AI developer tooling capabilities for users, and to have appropriate primitives for high performance, better scalability, and easier administration. That evolution applies to the compute services, relevant storage and the connectivity between them. W&B plans to use the operator to roll out such improvements to users across deployment types.

:::info
W&B uses the operator to deploy and manage Dedicated Cloud instances on AWS, GCP and Azure public clouds.
:::

## Reasons for the Architecture Shift
Historically, the W&B application was deployed as a single deployment and pod within a Kubernetes Cluster or a single Docker container. We have always recommended externalizing the Database and Object Store to decouple state from the application, especially in production environments.

As the application grew, the need to evolve from a monolithic container to a distributed system became apparent. This change facilitates backend logic handling and seamlessly introduces **_in-kubernetes_** infrastructure capabilities. It also supports deploying new services essential for additional features that W&B relies on.

Previously, any Kubernetes-related changes required updating the [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb), ensuring compatibility across cloud providers, configuring necessary Terraform variables, and executing a terraform apply for each backend or Kubernetes-level change. This process was not scalable and placed a significant burden on our support staff to assist customers with upgrades.

The solution was to implement an **_Operator_** that connects to a central [deploy.wandb.ai](https://deploy.wandb.ai) server to request the latest specification changes for a given **_Release Channel_** and apply them. Updates will be received as long as the license is valid. Helm was chosen as both the deployment mechanism for our operator and the means for the operator to handle all configuration templating of the W&B Kubernetes stack; Helmception.

## How it works
The operator can be installed from [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator). This installation creates a deployment called `controller-manager` and utilizes a **_Custom Resource_** definition named `weightsandbiases.apps.wandb.com` (shortName: `wandb`), which takes a single `spec` and applies it to the cluster:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

The `controller-manager` installs [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) based on the spec of the **_Custom Resource_**, **_Release Channel_**, and a **_User Defined Config_**. This hierarchy allows for maximum configuration flexibility at the user end and enables W&B to release new images, configurations, features, and Helm updates without the customer having to do anything.

For a detailed description of the Specification Hierarchy please see [Configuration Specification Hierarchy](#configuration-specification-hierarchy) and for the configuration options, please see Configuration Reference.

For more general information about the operator, see [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) in the Kubernetes documentation.

## Configuration Specification Hierarchy
In our operator model, configuration specifications follow a hierarchical model where higher-level specifications override lower-level ones. Here’s how it works:
- **Release Channel Values**: This base level configuration sets default values and configurations based on the **_Release Channel_** set by W&B for the deployment.
- **User Input Values**: Users can override the default settings provided by the Release Channel Spec through the System Console.
- **Custom Resource Values**: The highest level of specification, which comes from the user. Any values specified here will override both the User Input and Release Channel specifications. For a detailed description of the configuration options, please see [Configuration Reference](#configuration-reference).

This hierarchical model ensures that configurations are flexible and customizable to meet varying needs while maintaining a manageable and systematic approach to upgrades and changes.

## Requirements to use the W&B Kubernetes Operator
To successfully deploy W&B with  the W&B Kubernetes Operator, the following requirements must be met:
* Egress to the following endpoints during installation and during runtime:
    * deploy.wandb.ai
    * docker.io
    * quay.io
    * gcr.io
* A Kubernetes cluster at least version 1.28 with a deployed, configured and fully functioning Ingress controller (e.g. Contour, Nginx).
* Externally hosted and running MySQL 8.0 database.
* Object Storage (AWS S3, Azure Cloud Storage, Google Cloud Storage, or any S3-compatible storage service) with CORS support.
* A valid W&B Server license.

Please see [this](./self-managed/bare-metal) guide for a detailed explanation on how to set up and configure a self-managed installation.

**Optional**

Depending on the installation method, you might need to meet the following requirements:
* Kubectl installed and configured with the correct Kubernetes cluster context.
* Helm installed.

# Air-gapped installations
Please note that air-gapped installations are currently not supported by the W&B Kubernetes Operator. W&B is working on a utility to deploy W&B with the Operator in air-gapped environments.

InfoBox: We are actively working on adding airgapped installation support. For a status update, please reach out to [Customer Support](mailto:support@wandb.com) or your W&B team.

# Deploy W&B Server application
This section describes different ways to deploy the W&B Kubernetes operator. 
:::note
The W&B Operator will become the default installation method for W&B Server. Other methods will be deprecated in the future.
:::

**Choose one of the following:**
- If you have provisioned all required external services and want to deploy W&B onto Kubernetes with Helm CLI, please continues [here](#deploy-wb-with-helm-cli).
- If you prefer managing infrastructure and the W&B Server with Terraform, please continue [here](#deploy-wb-with-helm-terraform-module).
- If you want to utilize the W&B Cloud Terraform Modules, please continues [here](#deploy-wb-with-wb-cloud-terraform-modules).

## Deploy W&B with Helm CLI
W&B provides a Helm Chart to deploy the W&B Kubernetes Operator to a Kubernetes cluster. This approach allows you to deploy W&B Server via Helm CLI or a continuous delivery tool like ArgoCD. Please make sure that the above mentioned requirements are in place.

Follow those steps to install the W&B Kubernetes Operator via Helm CLI:

**Step 1: Add the wandb Helm Repository**

W&B has made the W&B Helm chart available via the W&B Helm repository. Add the repo with the following commands:

```console
helm repo add wandb https://charts.wandb.ai
helm repo update
```

**Step 2: Install the Operator**

To install the Operator on the Kubernetes cluster, execute this command:

```console
helm upgrade --install operator wandb/operator
```

**Step 3: Configuring the W&B operator custom resource to trigger the W&B Server installation**

Create an operator.yaml file to customize the W&B Operator deployment, specifying your custom configuration. See [Configuration Reference](#configuration-reference) for details.

Once you have the specification YAML created and filled with your values, run the following and the operator will apply the configuration and install the W&B Server application based on your configuration.

```console
kubectl apply -f operator.yaml
```

Wait until the deployment is completed and verify the installation. This will take a few minutes.

**Step 4: Verify the installation**

Access the new installation via the browser and create the first admin user account. When this is done, follow the verification steps as outlined [here](#verify-the-installation)


## Deploy W&B with Helm Terraform Module

This method allows for customized deployments tailored to specific requirements, leveraging Terraform's infrastructure-as-code approach for consistency and repeatability. The official W&B Helm-based Terraform Module that is located here: [terraform-helm-wandb](https://registry.terraform.io/modules/wandb/wandb/helm/latest). 

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

Please note that the configuration options are the same as described in [Configuration Reference](#configuration-reference), but that the syntax has to be written in the HashiCorp Configuration Language (HCL). The W&B custom resource definition will be created by the Terraform module.

To see how Weights&Biases themselves use the Helm Terraform module to deploy “Dedicated Cloud” installations for customers, please follow those links:
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

## Deploy W&B with W&B Cloud Terraform modules

W&B provides a set of Terraform Modules for AWS, GCP and Azure. Those modules deploy entire infrastructures including Kubernetes clusters, load balancers, MySQL databases and so on as well as the W&B Server application. The W&B Kubernetes Operator is already pre-baked with those official W&B cloud-specific Terraform Modules with the following versions:

| Terraform Registry | Source Code                  | Version |
| -------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest)  | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)            | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)               | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

This integration ensures that W&B Kubernetes Operator is ready to use for your instance with minimal setup, providing a streamlined path to deploying and managing W&B Server in your cloud environment.

For a detailed description on how to use these modules, please refer to this [section](./hosting-options/self-managed#deploy-wb-server-within-self-managed-cloud-accounts) to self-managed installations section in the docs.

## Verify the installation

To verify the installation, W&B recommends using the wandb CLI. The verify command will execute several tests that verify all components and configurations. This step assumes that the first admin user account has been created via the browser.

Please follow these steps to verify the installation:

**Step 1: Install the CLI**

```console
pip install wandb
```

**Step 2: Log in to W&B**
```console
wandb login --host=https://YOUR_DNS_DOMAIN
```

Example:
```console
wandb login --host=https://wandb.company-name.com
```

**Step 3: Start the verification**

```console
wandb verify
```

A successful execution and fully working W&B deployment will show this output:

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

## Accessing the W&B Management Console
The W&B Kubernetes Operator Management comes with a management console. It is located at ${HOST_URI}/console, e.g. https://wandb.company-name.com/console.

There are two ways to login to it:

**Option 1**

**Step 1: Open the W&B application in the browser and login**

Login to the W&B application via ${HOST_URI}/, e.g. https://wandb.company-name.com/

**Step 2: Access the console**

Click on the icon in the top right corner and then on “System console”. Please note that only users with admin privileges will see the “System console” entry.

![](/images/hosting/access_system_console_via_main_app.png)


**Option 2**

**Step 1: Open console application in browser**

Open the above described URL in the browser and you will be presented with this login screen:

![](/images/hosting/access_system_console_directly.png)

**Step 2: Retrieve password**

The password is stored as a Kubernetes secret and is being generated as part of the installation. To retrieve it, execute the following command:

```console
kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
```

Copy the password to the clipboard.

**Step 3: Login to the console**

Paste the copied password to the textfield “Enter password” and click login.

:::note
The second option is recommended for troubleshooting only should the main application not be accessible. 
:::


## Update the W&B Kubernetes Operator
This section describes how to update the operator itself. Please note that this will not update the W&B server application and should you be using a very old version of the Helm chart (non-operator based), please first migrate to the new version of the chart (operator based) as described [here](#migrate-self-managed-instances-to-wb-kubernetes-operator). To update the W&B server application, please continue [here](#update-the-wb-server-application):

Please follow these steps:

**Step 1: Update the repo**

```console
helm repo update
```

**Step 2: Update the Helm chart itself**
 
```console
helm upgrade --reuse-values
```

## Update the W&B Server application
Customers do not have to update the W&B server application by themselves anymore. The operator will take care of the update as soon as Weights&Biases is releasing a new version of the software.

## Migrate self-managed instances to W&B Operator
W&B recommends that you use the operator if you self-manage your W&B Server instance. That would enable W&B to roll out the newer services and products to your instance more seamlessly, and provide better troubleshooting and support.

:::note
The W&B Operator will become the default installation method for W&B Server. In future, W&B will deprecate deployment mechanisms that do not use the operator. Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions.
:::

**Step 1: Collect data of current installation**

At first we need to identify the current installation method as further steps will depend on that:
- If you used the official W&B Cloud Terraform Modules, please jump to the according section and continue there:
  - [AWS](#migrate-to-operator-based-aws-terraform-modules)
  - [GCP](#migrate-to-operator-based-gcp-terraform-modules)
  - [Azure](#migrate-to-operator-based-azure-terraform-modules)
- If you used the [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb), please continue [here](#migrate-to-operator-based-helm-chart).
- If you used the [W&B Non-Operator Helm chart via Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest), please continue [here](#migrate-to-operator-based-terraform-helm-chart).
- If you created the Kubernetes resources via manifest(s), please continue [here](#migrate-to-operator-based-helm-chart).


### Migrate to Operator-based AWS Terraform Modules

For a detailed description of the migration process, please continue [here](self-managed/aws-tf#migrate-to-operator-based-aws-terraform-modules).

### Migrate to Operator-based GCP Terraform Modules

:::note
This section of the documentation is currently being worked on. Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions or need assistance.
:::

### Migrate to Operator-based Azure Terraform Modules

:::note
This section of the documentation is currently being worked on. Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions or need assistance.
:::

### Migrate to Operator-based Helm chart

Follow these steps to migrate to the Operator-based Helm chart:

**Step 1: Get the current W&B configuration**

If W&B was deployed with an non-operator-based version of the Helm chart, please export the values like this:

```console
helm get values wandb > operator.yaml
```

If W&B was deployed with Kubernetes manifests, please export the values like this:

```console
kubectl get deployment wandb -o yaml
```

In both ways you should now have all the configuration values which are needed for the next step. 

**Step 2: Create operator.yaml**

Create a file called operator.yaml by following the format described in the [Configuration Reference](#configuration-reference). Use the values from step 1.

**Step 3: Scale the current deployment to 0 pods**

This step is stopping the current deployment.

```console
kubectl scale --replicas=0 deployment wandb
```

**Step 4: Update the Helm chart repo**

```console
helm repo update
```

**Step 5: Install the new Helm chart**

```console
helm upgrade --install operator wandb/operator
```

**Step 6: Configure the new helm chart and trigger W&B application deployment**

Apply the new configuration.
```console
kubectl apply -f operator.yaml
```

The deployment will take a few minutes to complete.

**Step 7: Verify the installation**

Make sure that everything works by following the steps in [Verify the installation](#verify-the-installation).

**Step 8: Remove to old installation**

Uninstall the old helm chart or delete the resources that were created via manifests.

### Migrate to Operator-based Terraform Helm chart

Follow these steps to migrate to the Operator-based Helm chart:


**Step 1: Prepare Terraform config**
- Replace the Terraform code from the old deployment in your Terraform config with the one that is described here: Link to Deploy w… Terraform code
- Set the same variables as before. Should you have a .tfvars file, leave it unchanged.

**Step 2: Execute Terraform run**

Execute terraform init, plan and apply

**Step 3: Verify the installation**

Make sure that everything works by following the steps in [Verify the installation](#verify-the-installation).

**Step 4: Remove to old installation**

Uninstall the old helm chart or delete the resources that were created via manifests.



## Configuration Reference

This section describes the configuration options for W&B Server application. The application receives its configuration as custom resource definition named [WeightsAndBiases](#how-it-works). Many configuration options have been exposed via below configuration, some need to be set as environment variables (see below: Additional Environment Variables). 

The documentation has two lists of environment variables: Link1 and Link2. Only use environment variables if the configuration option that you need has not yet been exposed via Helm Chart.

The W&B Kubernetes operator configuration file for a production deployment requires the following contents:

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

This YAML file defines the desired state of your Weights & Biases deployment, including the version, environment variables, external resources like databases, and other
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
      extraEnv:
        GLOBAL_ENV: "example"
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
  # example host name, please replace with your own
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
```

**GCP**
```yaml
global:
  bucket:
    name: abc-wandb-moving-pipefish
    provider: gcs
```

**Azure**
ToDO: fix
```yaml
global:
  bucket:
    # az, s3, gcs
    provider: "s3"
    name: ""
    path: ""
    region: ""
    kmsKey: ""
    secretKey: ""
    accessKey: ""
```

**Other providers (Minio, Ceph, etc.)**

These are all available configuration options for object storage:

```yaml
global:
  bucket:
    # az, s3, gcs
    provider: "s3"
    name: ""
    path: ""
    region: ""
    kmsKey: ""
    secretKey: ""
    accessKey: ""
```

Set *kms_key* to **null** for all other providers.

### MySQL

```yaml
# disable in-chart MySQL
mysql:
  install: false

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
  # Example license, please replace with your own
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

### Ingress
**Without TLS**

```yaml
global:
# IMPORTANT: Ingress is on the same level in the YAML as ‘global’ (not a child)
ingress:
  class: ""
  annotations:
    {}
    # kubernetes.io/ingress.class: nginx
    # kubernetes.io/tls-acme: "true"
  tls: []
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

### External Redis

```yaml
redis:
  install: false

global
  redis:
    host: ""
    port: 6379
    password: ""
    parameters: {}
    caCert: ""
```

Alternatively via redis password in a Kubernetes secret:

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


## FAQ

#### How to get the  W&B Operator Console password
Please see [Accessing the W&B Kubernetes Operator Management Console](#accessing-the-wb-management-console).


#### How to access the W&B Operator Console if Ingress doesn’t work

Execute the following command on a host that can reach the Kubernetes cluster:

```console
kubectl port-forward svc/wandb-console 8082
```

Access the console in the browser via https://localhost:8082/console.

Please see [Accessing the W&B Kubernetes Operator Management Console](#accessing-the-wb-management-console) on how to get the password (Option 2).

#### How to view W&B Server logs

The application pod is named **wandb-app-xxx**.

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```