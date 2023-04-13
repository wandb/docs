# Secure storage connector

The team-level secure storage connector allows teams within W&B to utilize a separate cloud file storage bucket from the rest of the W&B instance. This provides greater data access control and data isolation for teams with highly sensitive data or strict compliance requirements.

:::info
This feature is currently only available for Google Cloud Platform and Amazon Web Services. To request a license to enable this feature, email contact@wandb.com.
:::

A cloud storage bucket can be configured only once for a team at the time of team creation. Select **External Storage** when you create a team tp configure a cloud storage bucket. You can configure a cloud storage bucket once the bucket is provisioned. Select your provider and fill out your bucket name and storage encryption key ID, if applicable, and select **Create team bucket**.

An error or warning will appear at the bottom of the page if there are issues accessing the bucket or the bucket has invalid settings.

![](/images/hosting/prod_setup_secure_storage.png)

Only system administrators have the permissions to configure the secure storage connector. The same cloud storage bucket can be used amongst multiple teams by selecting an existing cloud storage bucket from the dropdown.
