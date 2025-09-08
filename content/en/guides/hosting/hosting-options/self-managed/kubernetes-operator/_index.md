---
description: Deploy W&B Platform with Kubernetes Operator
title: Run W&B Server on Kubernetes
menu:
  default:
    identifier: kubernetes-operator
    parent: self-managed
weight: 2
url: guides/hosting/operator
---

Use the W&B Kubernetes Operator to simplify deploying, administering, troubleshooting, and scaling your W&B Server deployments on Kubernetes. You can think of the operator as a smart assistant for your W&B instance.

The W&B Server architecture and design continuously evolves to expand AI developer tooling capabilities, and to provide appropriate primitives for high performance, better scalability, and easier administration. That evolution applies to the compute services, relevant storage and the connectivity between them. To help facilitate continuous updates and improvements across deployment types, W&B uses a Kubernetes operator.

{{% alert %}}
W&B uses the operator to deploy and manage Dedicated Cloud instances on AWS, GCP and Azure public clouds.
{{% /alert %}}

For more information about Kubernetes operators, see [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) in the Kubernetes documentation. For more details about the design of the W&B Kubernetes Operator, refer to [Appendix: Conceptual details]({{< relref "#appendix-conceptual-details" >}}).

## Requirements

{{% readfile "/_includes/server-kubernetes-requirements.md" %}}

{{% alert title="Bare-metal or air-gapped deployments" %}}
This page shows how to deploy W&B in a cluster deployed in a public cloud or in a datacenter connected to the internet.

- For detailed instructions for a bare-metal deployment, refer to [Hosting W&B Server on on-premises infrastructure]({{< relref "../bare-metal.md" >}}.
- For detailed instructions for an air-gapped deployment without internet access, refer to [Deploy W&B in airgapped environment with Kubernetes]({{< relref "operator-airgapped.md" >}})
{{% /alert %}}


See the [bare-metal installation guide]({{< relref "../bare-metal.md" >}}) for a detailed explanation on how to set up and configure a self-managed installation.

If necessary, Install `kubectl` and configure it with the correct Kubernetes cluster context.


## Deploy W&B Server application
This section describes different ways to use the W&B Kubernetes Operator to deploy W&B.
{{% alert %}}
The W&B Operator is the default and recommended installation method for W&B Server.
{{% /alert %}}

W&B provides a Helm Chart to deploy the W&B Kubernetes operator to a Kubernetes cluster. This approach allows you to deploy W&B Server with Helm CLI or a continuous delivery tool like ArgoCD. Make sure that the above mentioned requirements are in place.

To deploy W&B Server, you can use Terraform to deploy the Helm chart, or you can use Helm directly. Select a tab to continue.

{{< tabpane text=true >}}
  {{% tab header="Terraform" %}}
This section shows how to use Terraform to deploy W&B to your own infrastructure, such as a datacenter, using Terraform. To deploy to your own AWS, Azure, or GCP tenant instead, refer to [Deploy W&B with W&B Cloud Terraform modules]({{< relref "#deploy-wb-with-wb-cloud-terraform-modules" >}}).

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
    }
  }
}
```

Note that the configuration options are the same as described in [Configuration Reference]({{< relref "#configuration-reference-for-wb-operator" >}}), but that the syntax has to follow the HashiCorp Configuration Language (HCL). The Terraform module creates the W&B custom resource definition (CRD).

To see how W&B themselves use the Helm Terraform module to deploy Dedicated Cloud installations for customers, follow these links:
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### Deploy W&B with W&B Cloud Terraform modules
This section shows how to use Terraform to deploy W&B in your own AWS, Azure, or GCP tenant. To deploy W&B Self-Hosted on your own infrastructure, such as in a datacenter, refer to [Deploy W&B with W&B Cloud Terraform modules]({{< relref "#deploy-wb-with-wb-cloud-terraform-modules" >}}) instead.

W&B provides a set of Terraform Modules for AWS, GCP and Azure. Those modules deploy entire infrastructures including Kubernetes clusters, load balancers, MySQL databases and so on as well as the W&B Server application. The W&B Kubernetes Operator is already pre-baked with those official W&B cloud-specific Terraform Modules with the following versions:

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

This integration ensures that W&B Kubernetes Operator is ready to use for your instance with minimal setup, providing a streamlined path to deploying and managing W&B Server in your cloud environment.

For a detailed description on how to use these modules, refer to this [section]({{< relref "../#deploy-wb-server-within-self-managed-cloud-accounts" >}}) to self-managed installations section in the docs.
  {{% /tab %}}
  {{% tab header="Helm" %}}
Follow these seteps to install the W&B Operator with Helm and then use the Operator to deploy W&B Server.

### Install the W&B Operator {#install-wb-operator-helm}
To install and start the Kubernetes Operator with Helm:

1. Add the W&B Helm repository. The W&B Helm chart is available in the W&B Helm repository:
    ```shell
    helm repo add wandb https://charts.wandb.ai
    helm repo update
    ```
1. Install and start the Operator on a Kubernetes cluster.

    - To start the Operator with the default configuration:

        ```shell
        helm upgrade --install install operator wandb/operator
        ```

    - To start the Operator with a `values.yaml` override:

        ```shell
        helm upgrade --install operator wandb/operator -f values.yaml 
        ```

        Refer to the [example `values.yaml`](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml).

    Wait until the deployment completes. This takes a few minutes.
### Deploy W&B Server with Helm  {#deploy-wb-helm}

After [installing and starting the Kubernetes Operator]({{< relref "#install-the-operator-with-helm" >}}), follow these steps to deploy W&B:

1. To configure your deployment, create the `WeightsAndBiases` spec in a file named `operator.yaml`, based on the [W&B config reference]({{< relref "#configuration-reference-for-wb-server" >}}).
1. Deploy the CR using `kubectl apply`:
    ```shell
    kubectl apply -f operator.yaml
    ```
1. To verify the installation using the web UI, create the first admin user account, then follow the verification steps outlined in [Verify the installation]({{< relref "#verify-the-installation" >}}).
  {{% /tab %}}
{{< /tabpane >}}

## Verify the installation

To verify the installation, W&B recommends using the [W&B CLI]({{< relref "/ref/cli/" >}}). The verify command executes several tests that verify all components and configurations. 

{{% alert %}}
This step assumes that the first admin user account is created with the browser.
{{% /alert %}}

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
The W&B Kubernetes operator comes with a management console. It is located at `${HOST_URI}/console`, for example `https://wandb.company-name.com/console`.

There are two ways to log in to the management console:

{{< tabpane text=true >}}
{{% tab header="Option 1 (Recommended)" value="option1" %}}
1. Open the W&B application in the browser and login. Log in to the W&B application with `${HOST_URI}/`, for example `https://wandb.company-name.com/`
2. Access the console. Click on the icon in the top right corner and then click **System console**. Only users with admin privileges can see the **System console** entry.

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="System console access" >}}
{{% /tab %}}

{{% tab header="Option 2" value="option2"%}}
{{% alert %}}
W&B recommends you access the console using the following steps only if Option 1 does not work.
{{% /alert %}}

1. Open console application in browser. Open the above described URL, which redirects you to the login screen:
    {{< img src="/images/hosting/access_system_console_directly.png" alt="Direct system console access" >}}
2. Retrieve the password from the Kubernetes secret that the installation generates:
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    Copy the password.
3. Login to the console. Paste the copied password, then click **Login**.
{{% /tab %}}
{{< /tabpane >}}

## Update the W&B Kubernetes operator
This section describes how to update the W&B Kubernetes operator. 

{{% alert %}}
* Updating the W&B Kubernetes operator does not update the W&B server application.
* See the instructions [here]({{< relref "#migrate-self-managed-instances-to-wb-operator" >}}) if you use a Helm chart that does not use the W&B Kubernetes operator before you follow the proceeding instructions to update the W&B operator.
{{% /alert %}}

Select a tab to continue.

{{< tabpane text=true >}}
  {{% tab header="Terraform" %}}
To update the W&B Kubernetes operator using Terraform:

1. Update the `operator_chart_version` variable in your Terraform configuration to the desired version. To use the latest version:
    ```hcl
    module "wandb" {
      source  = "wandb/wandb/helm"
      version = "~> 1.0"
      
      operator_chart_version = "latest"
      # ... other configuration ...
    }
    ```

2. Run Terraform commands to apply the update:
    ```shell
    terraform init -upgrade
    terraform plan
    terraform apply
    ```

The Terraform module will automatically update the operator Helm release to the specified version while preserving your existing configuration.

{{% alert title="Note" %}}
If you're using a specific version instead of "latest", check the [W&B Helm Charts repository](https://github.com/wandb/helm-charts/tree/main/charts/operator) for available versions.
{{% /alert %}}
  {{% /tab %}}
  
  {{% tab header="Helm" %}}
Copy and paste the code snippets below into your terminal. 

1. First, update the repo with [`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/):
    ```shell
    helm repo update
    ```

2. Next, update the Helm chart with [`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/):
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```
  {{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
You no longer need to update W&B Server application if you use the W&B Kubernetes operator.

The operator automatically updates your W&B Server application when a new version of the software of W&B is released.
{{% /alert %}}


## Migrate Self-Managed instances to W&B Operator
The proceeding section describe how to migrate from self-managing your own W&B Server installation to using the W&B Operator to do this for you. 

{{% alert %}}
The W&B Operator is the default and recommended installation method for W&B Server. Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions.
{{% /alert %}}

Select a tab to continue.

{{< tabpane text=true >}}
  {{% tab header="Terraform" %}}
This section covers migration deployments using:
- W&B Cloud Terraform Modules (AWS, GCP, Azure)
- W&B Non-Operator Helm chart deployed with Terraform

### Self-Managed Terraform deployments without the Kubernetes operator{#migrate-non-operator-terraform}

For a detailed description of the migration process, see the [AWS Terraform deployment guide]({{< relref "../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" >}}).

#### Migrate to Operator-based GCP Terraform Modules

Contact [support](mailto:support@wandb.com) or your W&B account team for details or assistance.

#### Migrate to Operator-based Azure Terraform Modules

Contact [support](mailto:support@wandb.com) or your W&B account team for details or assistance.

### Self-managed Terraform deployments without Helm

Follow these steps to migrate to the Operator-based Terraform Helm module:

1. **Prepare Terraform config**: Replace the Terraform code from the old deployment in your Terraform config with the one that is described [here]({{< relref "#deploy-wb-with-helm-terraform-module" >}}). Set the same variables as before. Do not change `.tfvars` file if you have one.

2. **Execute Terraform run**: Execute terraform commands:
    ```shell
    terraform init
    terraform plan
    terraform apply
    ```

3. **Verify the installation**: Make sure that everything works by following the steps in [Verify the installation]({{< relref "#verify-the-installation" >}}).

4. **Remove old installation**: The Terraform apply should handle the migration automatically, replacing the old resources with the new operator-based ones.
  {{% /tab %}}
  
  {{% tab header="Helm" %}}
This section covers migration for:
- Deployments that use Helm but do not use the W&B Operator.
- Kubernetes manifest deployments.

### Self-Managed Helm deployments without the Kubernetes operator  {#migrate-non-operator-helm}

Follow these steps to migrate to the Operator-based Helm chart:

1. Get the current W&B configuration. If W&B was deployed with an non-operator-based version of the Helm chart, export the values like this:
    ```shell
    helm get values wandb
    ```
    If W&B was deployed with Kubernetes manifests, export the values like this:
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    You now have all the configuration values you need for the next step. 

2. Create a file called `operator.yaml`. Follow the format described in the [Configuration Reference]({{< relref "#configuration-reference-for-wb-operator" >}}). Use the values from step 1.

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
    The deployment takes a few minutes to complete.

7. Verify the installation. Make sure that everything works by following the steps in [Verify the installation]({{< relref "#verify-the-installation" >}}).

8. Remove to old installation. Uninstall the old helm chart or delete the resources that were created with manifests.
  {{% /tab %}}
{{< /tabpane >}}


## Configuration Reference for W&B Server

This section describes the configuration options for W&B Server application. The application receives its configuration as custom resource definition named [WeightsAndBiases]({{< relref "#how-it-works" >}}). Some configuration options are exposed with the below configuration, some need to be set as environment variables.

The documentation has two lists of environment variables: [basic]({{< relref "/guides/hosting/env-vars.md" >}}) and [advanced]({{< relref "/guides/hosting/iam/advanced_env_vars.md" >}}). Only use environment variables if the configuration option that you need are not exposed using Helm Chart.

This YAML file defines the desired state of your W&B deployment, including the version, environment variables, external resources like databases, and other necessary settings.

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
```

Find the full set of values in the [W&B Helm repository](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml). **Change only those values you need to override**.

### Complete example 
This example configuration deploys W&B to GCP Anthos using Google Cloud Storage:

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
```

### Host
```yaml
 # Provide the FQDN with protocol
global:
  # example host name, replace with your own
  host: https://wandb.example.com
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

For other S3 compatible providers, set the bucket configuration as follows:
```yaml
global:
  bucket:
    # Example values, replace with your own
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

For S3-compatible storage hosted outside of AWS, `kmsKey` must be `null`.

To reference `accessKey` and `secretKey` from a secret:
```yaml
global:
  bucket:
    # Example values, replace with your own
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    secret:
      secretName: bucket-secret
      accessKeyName: ACCESS_KEY
      secretKeyName: SECRET_KEY
```

### MySQL

```yaml
global:
   mysql:
     # Example values, replace with your own
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     password: 8wtX6cJH...ZcUarK4zZGjpV 
```

To reference the `password` from a secret:
```yaml
global:
   mysql:
     # Example values, replace with your own
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     passwordSecret:
       name: database-secret
       passwordKey: MYSQL_WANDB_PASSWORD
```

### License

```yaml
global:
  # Example license, replace with your own
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

To reference the `license` from a secret:
```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

To identify the ingress class,  see this FAQ [entry]({{< relref "#how-to-identify-the-kubernetes-ingress-class" >}}).

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
The subsystems "app" and "parquet" run under the specified service account. The other subsystems run under the default service account.

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

You can specify service accounts on different subsystems such as app, parquet, console, and others:

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

To reference the `password` from a secret:

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
      # Only include if your IdP requires it.
      authMethod: ""
      issuer: ""
```

`authMethod` is optional. 

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
    MIIBxTCCAWugAwIB.......................qaJcwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
    SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
    aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
    -----END CERTIFICATE-----
```

CA certificates can also be stored in a ConfigMap:
```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

The ConfigMap must look like this:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-ca-certs
data:
  ca-cert1.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
  ca-cert2.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
```

{{% alert %}}
If using a ConfigMap, each key in the ConfigMap must end with `.crt` (for example, `my-cert.crt` or `ca-cert1.crt`). This naming convention is required for `update-ca-certificates` to parse and add each certificate to the system CA store.
{{% /alert %}}

### Custom security context

Each W&B component supports custom security context configurations of the following form:

```yaml
pod:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    runAsGroup: 0
    fsGroup: 1001
    fsGroupChangePolicy: Always
    seccompProfile:
      type: RuntimeDefault
container:
  securityContext:
    capabilities:
      drop:
        - ALL
    readOnlyRootFilesystem: false
    allowPrivilegeEscalation: false 
```

{{% alert %}}
The only valid value for `runAsGroup:` is `0`. Any other value is an error.
{{% /alert %}}


For example, to configure the application pod, add a section `app` to your configuration:

```yaml
global:
  ...
app:
  pod:
    securityContext:
      runAsNonRoot: true
      runAsUser: 1001
      runAsGroup: 0
      fsGroup: 1001
      fsGroupChangePolicy: Always
      seccompProfile:
        type: RuntimeDefault
  container:
    securityContext:
      capabilities:
        drop:
          - ALL
      readOnlyRootFilesystem: false
      allowPrivilegeEscalation: false 
```

The same concept applies to `console`, `weave`, `weave-trace` and `parquet`.

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
  MIIBxTCCAWugAwIB.......................qaJcwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
  SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
  aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
  -----END CERTIFICATE-----
```

CA certificates can also be stored in a ConfigMap:
```yaml
caCertsConfigMap: custom-ca-certs
```

The ConfigMap must look like this:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-ca-certs
data:
  ca-cert1.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
  ca-cert2.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
```

{{% alert %}}
Each key in the ConfigMap must end with `.crt` (e.g., `my-cert.crt` or `ca-cert1.crt`). This naming convention is required for `update-ca-certificates` to parse and add each certificate to the system CA store.
{{% /alert %}}

## FAQ

### What is the purpose/role of each individual pod?
* **`wandb-app`**: the core of W&B, including the GraphQL API and frontend application. It powers most of our platform’s functionality.
* **`wandb-console`**: the administration console, accessed via `/console`. 
* **`wandb-otel`**: the OpenTelemetry agent, which collects metrics and logs from resources at the Kubernetes layer for display in the administration console.
* **`wandb-prometheus`**: the Prometheus server, which captures metrics from various components for display in the administration console.
* **`wandb-parquet`**: a backend microservice separate from the `wandb-app` pod that exports database data to object storage in Parquet format.
* **`wandb-weave`**: another backend microservice that loads query tables in the UI and supports various core app features.
* **`wandb-weave-trace`**: a framework for tracking, experimenting with, evaluating, deploying, and improving LLM-based applications. The framework is accessed via the `wandb-app` pod.

### How to get the  W&B Operator Console password
See [Accessing the W&B Kubernetes Operator Management Console]({{< relref "#access-the-wb-management-console" >}}).


### How to access the W&B Operator Console if Ingress doesn’t work

Execute the following command on a host that can reach the Kubernetes cluster:

```console
kubectl port-forward svc/wandb-console 8082
```

Access the console in the browser with `https://localhost:8082/` console.

See [Accessing the W&B Kubernetes Operator Management Console]({{< relref "#access-the-wb-management-console" >}}) on how to get the password (Option 2).

### How to view W&B Server logs

The application pod is named **wandb-app-xxx**.

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### How to identify the Kubernetes ingress class

You can get the ingress class installed in your cluster by running

```console
kubectl get ingressclass
```

## Appendix: Conceptual details

This section provides more details about the design of the W&B Kubernetes operator.

### Reasons for the architecture shift
Historically, the W&B application was deployed as a single deployment and pod within a Kubernetes Cluster or a single Docker container. W&B has, and continues to recommend, to externalize the Database and Object Store. Externalizing the Database and Object store decouples the application's state.

As the application grew, the need to evolve from a monolithic container to a distributed system (microservices) was apparent. This change facilitates backend logic handling and seamlessly introduces built-in Kubernetes infrastructure capabilities. Distributed systems also supports deploying new services essential for additional features that W&B relies on.

Before 2024, any Kubernetes-related change required manually updating the [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform module. Updating the Terraform module ensures compatibility across cloud providers, configuring necessary Terraform variables, and executing a Terraform apply for each backend or Kubernetes-level change. 

This process was not scalable since W&B Support had to assist each customer with upgrading their Terraform module.

The solution was to implement an operator that connects to a central [deploy.wandb.ai](https://deploy.wandb.ai) server to request the latest specification changes for a given release channel and apply them. Updates are received as long as the license is valid. [Helm](https://helm.sh/) is used as both the deployment mechanism for the W&B operator and the means for the operator to handle all configuration templating of the W&B Kubernetes stack, Helm-ception.

### How it works
You can install the operator with helm or from the source. See [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator) for detailed instructions. 

The installation process creates a deployment called `controller-manager` and uses a [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) definition named `weightsandbiases.apps.wandb.com` (shortName: `wandb`), that takes a single `spec` and applies it to the cluster:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

The `controller-manager` installs [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb) based on the spec of the custom resource, release channel, and a user defined config. The configuration specification hierarchy enables maximum configuration flexibility at the user end and enables W&B to release new images, configurations, features, and Helm updates automatically.

Refer to the [configuration specification hierarchy]({{< relref "#configuration-specification-hierarchy" >}}) and [configuration reference]({{< relref "#configuration-reference-for-wb-operator" >}}) for configuration options.

The deployment consists of multiple pods, one per service. Each pod's name is prefixed with `wandb-`.

### Configuration specification hierarchy
Configuration specifications follow a hierarchical model where higher-level specifications override lower-level ones. Here’s how it works:

- **Release Channel Values**: This base level configuration sets default values and configurations based on the release channel set by W&B for the deployment.
- **User Input Values**: Users can override the default settings provided by the Release Channel Spec through the System Console.
- **Custom Resource Values**: The highest level of specification, which comes from the user. Any values specified here override both the User Input and Release Channel specifications. For a detailed description of the configuration options, see [Configuration Reference]({{< relref "#configuration-reference-for-wb-operator" >}}).

This hierarchical model ensures that configurations are flexible and customizable to meet varying needs while maintaining a manageable and systematic approach to upgrades and changes.
