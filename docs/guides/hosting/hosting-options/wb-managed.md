# W&B managed hosting


## Public cloud


## Dedicated cloud
A managed, dedicated deployment on W&B's single-tenant infrastructure in your choice of cloud region.

W&B is hosted in a dedicated virtual private network on W&B's single-tenant AWS or GCP infrastructure in your choice of cloud region. Use our Secure Storage Connector to connect your data to a scalable data store hosted on your company's private cloud.

Talk to our sales team by reaching out to [contact@wandb.com](mailto:contact@wandb.com).

### Configuration

We can configure an object store on your behalf, in which case there is no additional configuration that you need to perform. However, we also enable you to connect your object store with our Secure Storage Connector feature. The Secure Storage Connector enables you to connect your own secure data storage to this dedicated cloud infrastructure. This enables you full control of your data with isolation guarantees, while relying on W&B's cloud operational expertise to reduce your support burden and manage your dedicated infrastructure.

A W&BDedicated Cloud with Secure Storage Connector enabled requires the following resources in your private cloud account:

* An object store (S3 or GCS)
* A KMS key (also called Cloud KMS)


To set up the environment, you must run a simple terraform and provide W&B with the generated output so that W&B can complete the configuration. The links to the terraform scripts and instructions to run them can be found below for each supported cloud provider.

### Amazon Web Services

The simplest way to configure W&B within AWS is to use our [official Terraform](https://github.com/wandb/terraform-aws-wandb). Detailed instructions can be found in the [AWS](private-cloud/aws) section. If instead you want to configure services manually you can find [instructions here](configuration.md#amazon-web-services).

### Google Cloud Platform

The simplest way to configure W&B within GCP is to use our [official Terraform](https://github.com/wandb/terraform-google-wandb). Detailed instructions can be found in the [GCP](private-cloud/gcp) section. If instead you want to configure services manually you can find [instructions here](configuration.md#google-cloud-platform).

### Microsoft Azure Cloud

The simplest way to configure W&B within Azure is to use our [official Terraform](https://github.com/wandb/terraform-azurerm-wandb). Detailed instructions can be found in the [Azure](private-cloud/azure) section. If instead you want to configure services manually you can find [instructions here](configuration.md#azure).
