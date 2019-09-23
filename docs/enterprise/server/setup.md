---
title: Setting up W&B Enterprise Server
sidebar_label: Setup
---

A W&B Enterprise Server is a self-contained virtual machine provisioned on your private cloud, a physical server, or developer workstation. See the following for instructions for how to provision a new instance.

## Amazon Web Services

Before you begin, make sure you have access to our AMI. You'll need to send us your AWS Account ID (visible at your [Account Settings page](https://console.aws.amazon.com/billing/home?#/account)) and desired region. W&B will share access to the W&B Enterprise Server AMI to your account and send you an AMI ID.

### Launch the Instance

Go to EC2 > Images > AMIs in the AWS Console, select "Private images" in the search type dropdown, and search for "wandb". Select the last created image that appears, and click "Launch".

* **Choose Instance Type**: Make sure to select a `m5.2xlarge` instance or larger. W&B requires at least 4 cores and 16GB of memory.
* **Configure Instance**: If you plan to use a cloud file backend (this is optional), make sure your instance has an IAM role that allows it to access S3 and subscribe to SQS.
* **Add Storage**: If you plan on using the instance disk for file storage, be sure to provision the EBS disk with enough storage. The default is 300GB.
* **Configure Security Group**: Ensure that port 80 on your instance is accessible to any machine from which you want to run machine learning jobs, or to any IP range from which you plan to use the W&B web interface.

After launching your instance, wait for it to boot. Your instance will spin up and be accessible at port 80 at its public IP.

Your instance is usable from boot, but for advanced options, [you may now proceed to configuring your instance.](/enterprise/server/config)

### Configuring instance availability

By default, your Enterprise Server serves the web interface and API on port 80 via unencrypted HTTP.

To add SSL, put your instance behind an Amazon Load Balancer and add a certificate, either by uploading it, or by using Amazon Certificate Manager.

To serve your instance from a hostname, configure your DNS nameservers to point towards the instance IP or Amazon Load Balancer.

If you are not serving your instance from a hostname, you should associate an Amazon Elastic IP with the machine so it remains accessible at a stable IP address.

## Google Cloud Platform

Before you begin, make sure you have access to our Compute Image.

### Launch the Instance

Go to Compute Engine > Images in the GCP console, and find the W&B image. Click "Create Instance".

* **Machine Type**: Make sure to select an `n2-standard-4` instance or larger. W&B requires at least 4 cores and 16GB of memory.
* **Identity and API Access**: If you plan on using a cloud file backend, be sure your instance service account has access to Google Storage and Pubsub.
* **Firewall**: Enable "Allow HTTP traffic".

After creating your instance, wait for it to boot. It will spin up and be accessible at port 80 at its public IP.

Your instance is usable from boot, but for advanced options, [you may now proceed to configuring your instance.](/enterprise/server/config)

### Configuring instance availability

By default, your Enterprise Server serves the web interface and API on port 80 via unencrypted HTTP.

To add SSL, put your instance behind a Load Balancer and add a certificate using the Google console.

To serve your instance from a hostname, configure your DNS nameservers to point towards the instance IP or Google Load Balancer.

If you are not serving your instance from a hostname, you should associate an Elastic IP with the machine so it remains accessible at a stable IP address.

## Microsoft Azure

## VMWare

## Virtualbox

Contact the W&B team to gain access to the OVA file for the W&B Enterprise Server.

Once you have the file, in Virtualbox, go to File > Import Appliance, and find the path of the downloaded archive.

When creating your system, ensure to allocate at least 4 CPUs and 16GB of RAM if you intend to use this system for production workloads.

Your W&B Server will be ready to use from moments of it booting up!

Once your VM is created, go to Settings > Network > Advanced > Port Forwarding to forward port 80 on the guest machine to any desired port on the host.

For advanced options, [you may now proceed to configuring your instance.](/enterprise/server/config)
