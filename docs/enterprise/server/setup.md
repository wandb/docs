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
* **Add Tags**: Name your instance if you like.
* **Configure Security Group**: Ensure that port 80 on your instance is exposed to any machine from which you want to run machine learning jobs, or from which you want to access the W&B web interface.

After launching your instance, wait for it to boot. Your instance will spin up and be accessible at port 80 at its public IP.

Your instance is usable from boot, but for advanced options, [you can now proceed to configuring your instance.](/enterprise/server/config)

### Configuring DNS and SSL

By default, your Enterprise Server serves the web interface and API on port 80 via unencrypted HTTP.

To add SSL, put your instance behind an Amazon Load Balancer and add a certificate, either by uploading it, or by using Amazon Certificate Manager.

To serve your instance from a hostname, configure your DNS nameservers to point towards the instance IP or Amazon Load Balancer.

## Google Cloud Platform

## Microsoft Azure

## VMWare

## Virtualbox
