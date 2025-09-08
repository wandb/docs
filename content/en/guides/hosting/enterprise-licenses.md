---
title: Enterprise Licenses
description: Learn about W&B Enterprise licenses, what features they include, and how to obtain and configure them.
displayed_sidebar: default
---

# Enterprise Licenses

An Enterprise license unlocks advanced features in Weights & Biases designed for organizations that need enhanced security, compliance, and administrative capabilities. This page provides a comprehensive overview of Enterprise licenses, including what features they enable, how to obtain them, and how to configure them for your deployment.

## What is an Enterprise License?

An Enterprise license provides access to W&B's most advanced features for:
- **Security**: Enhanced authentication, encryption, and access controls
- **Compliance**: Audit logging, HIPAA compliance options, and data governance
- **Administration**: Advanced user management, custom roles, and automation capabilities
- **Scalability**: Performance optimizations and dedicated support

Enterprise licenses are available for both cloud and self-managed deployments of W&B.

## Deployment Options

### W&B Dedicated Cloud

[W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/_index.md" >}}) deployments **automatically include** an Enterprise license. No additional configuration is required - all Enterprise features are available immediately upon provisioning.

**Key characteristics:**
- Single-tenant infrastructure
- Managed by W&B
- Enterprise features enabled by default
- Choice of cloud provider and region

### W&B Self-Managed

[W&B Self-Managed]({{< relref "/guides/hosting/hosting-options/self-managed/_index.md" >}}) deployments require obtaining and configuring an Enterprise license separately. Without an Enterprise license, Self-Managed deployments operate with limited features.

**Key characteristics:**
- Deployed on your own infrastructure
- Full control over data and deployment
- Enterprise license must be obtained and configured
- Supports air-gapped environments

## Enterprise Features

The following features require an Enterprise license:

### Security & Authentication
- **Single Sign-On (SSO)**: Integration with identity providers like Okta, Auth0, and others
- **SCIM Provisioning**: Automated user provisioning and deprovisioning
- **Service Accounts**: Non-human accounts for CI/CD and automation
- **IP Allowlisting**: Restrict access based on IP addresses (Self-Managed only)

### Access Management
- **Custom Roles**: Create fine-grained roles beyond the default Member/Admin/View-Only roles
- **View-Only Users**: Read-only access for stakeholders who need visibility without edit permissions
- **Team-level Access Controls**: Advanced permissions at the team level
- **Project-level Access Controls**: Granular permissions for individual projects

### Data & Compliance
- **Audit Logs**: Detailed logs of user actions for compliance and security monitoring
- **HIPAA Compliance**: Available with specific configuration options
- **Customer-Managed Encryption Keys (CMEK)**: Use your own encryption keys for data at rest
- **Bring Your Own Bucket (BYOB)**: Store artifacts in your own cloud storage (via Secure Storage Connector)
- **Data Retention Controls**: Configure custom data retention policies

### Operations & Administration
- **Automations**: Trigger workflows based on events (Pro and Enterprise for Cloud)
- **Advanced Organization Dashboard**: Enhanced visibility into organization usage
- **Priority Support**: Faster response times and dedicated support channels
- **MySQL Database Support**: Use MySQL instead of Postgres (Self-Managed only)
- **S3-compatible Storage**: Use S3 or compatible object storage (Self-Managed only)

### Performance & Scale
- **Dedicated Resources**: Guaranteed compute and storage resources (Dedicated Cloud)
- **Custom Rate Limits**: Higher API rate limits for enterprise workloads
- **Multi-region Deployment**: Deploy across multiple regions (Dedicated Cloud)

## Obtaining an Enterprise License

### For W&B Dedicated Cloud

Enterprise licenses are included automatically. To get started with Dedicated Cloud:

1. [Contact W&B Sales](https://wandb.ai/site/contact-sales) to discuss your requirements
2. Work with the W&B team to provision your dedicated instance
3. All Enterprise features will be enabled upon deployment

### For W&B Self-Managed

To obtain an Enterprise license for Self-Managed deployments:

1. **Request an Enterprise Trial**:
   - Visit the [Self-Managed Enterprise Trial form](https://wandb.ai/site/for-enterprise/self-hosted-trial)
   - Or contact your W&B account team

2. **For Production Licenses**:
   - Contact [W&B Sales](https://wandb.ai/site/contact-sales)
   - Discuss your deployment size and requirements
   - Receive your production license key

## Configuring Enterprise Licenses (Self-Managed Only)

Once you have obtained an Enterprise license key, configure it in your Self-Managed deployment:

### For Kubernetes Deployments (Helm)

1. Update your `values.yaml` file:
```yaml
wandb:
  license: "YOUR_LICENSE_KEY_HERE"
```

2. Upgrade your Helm release:
```bash
helm upgrade wandb wandb/wandb \
  --namespace wandb \
  --reuse-values \
  --set license=$LICENSE_KEY
```

### For Terraform Deployments

1. Update your Terraform variables:
```hcl
variable "license" {
  default = "YOUR_LICENSE_KEY_HERE"
}
```

2. Apply the Terraform configuration:
```bash
terraform apply -var="license=$LICENSE_KEY"
```

### For Docker Deployments

Set the license key as an environment variable:
```bash
docker run -d \
  -e LICENSE="YOUR_LICENSE_KEY_HERE" \
  -e OTHER_ENV_VARS \
  wandb/local
```

### Via the UI (Post-Deployment)

For existing deployments without environment variable configuration:

1. Log in as an admin user
2. Navigate to **System Settings**
3. Find the **License** section
4. Enter your new license key
5. Save the changes

## Verifying Your License

To verify that your Enterprise license is active:

1. **For Admins**: Navigate to the System Settings page
2. **Via API**: Check the `/system/settings` endpoint
3. **Feature Availability**: Try accessing an Enterprise-only feature like Custom Roles

## License Renewal

Enterprise licenses have expiration dates. To ensure uninterrupted service:

- **30 days before expiration**: W&B will contact you about renewal
- **Upon expiration**: Enterprise features will become unavailable
- **To renew**: Contact support@wandb.ai or your account team

## Common Issues

### License Not Recognized
- Verify the license key is correctly formatted (no extra spaces)
- Ensure the license hasn't expired
- Check that the license is set in the correct configuration location

### Features Not Available After Setting License
- Restart your W&B services after setting the license
- Verify the license includes the specific features you're trying to access
- Check system logs for any license validation errors

### License Expiration Warnings
- Monitor the System Settings page for expiration notifications
- Set up alerts for license expiration in your monitoring system
- Keep your account team contact information up to date

## Best Practices

1. **Store License Keys Securely**: Treat license keys as sensitive credentials
2. **Monitor Expiration**: Set up alerts 30-60 days before license expiration
3. **Document Configuration**: Keep records of where and how licenses are configured
4. **Test in Staging**: Validate license updates in a non-production environment first
5. **Plan for Renewals**: Budget and plan for license renewals in advance

## Getting Help

For assistance with Enterprise licenses:

- **Technical Issues**: Contact support@wandb.ai
- **License Procurement**: Contact your W&B account team or sales@wandb.ai
- **Feature Questions**: Consult this documentation or contact support

## Related Resources

- [W&B Pricing](https://wandb.ai/site/pricing/)
- [Self-Managed Installation Guide]({{< relref "/guides/hosting/hosting-options/self-managed/_index.md" >}})
- [Dedicated Cloud Overview]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/_index.md" >}})
- [Security & Compliance]({{< relref "/guides/hosting/data-security/_index.md" >}})