---
description: Manage your organization's billing settings
menu:
  default:
    parent: settings
title: Manage billing settings
weight: 20
---

Navigate to your user profile page and select your user icon on the top right corner. From the dropdown, choose **Billing**, or choose **Settings** and then select the **Billing** tab.

## Plan details

The **Plan details** section summarizes your organization's current plan, charges, limits, and usage.

- For details and a list of users, click **Manage users**.
- For details about usage, click **View usage**.
- Amount of storage your organization uses, both free and paid. From here, you can purchase additional storage and manage storage that is currently in use. Learn more about [storage settings]({{< relref "storage.md" >}}).

From here, you can compare plans or talk to Sales.

## Plan usage
This section visually summarizes current usage and displays upcoming usage charges. For detailed insights into usage by month, click **View usage** on an individual tile. To export usage by calendar month, team, or project, click **Export CSV**.

### Usage alerts

{{% alert %}}
Usage alerts are not available on the [Enterprise plan](https://wandb.ai/site/pricing/).
{{% /alert %}}

For organizations on paid plans, admins receive alerts via email **once per billing period** when certain thresholds are met, along with details about how to increase your organization's limits if you are a [billing admin]({{< relref "#billing-admin" >}}) and how to contact a billing admin otherwise. On the [Pro plan](https://wandb.ai/site/pricing/), only the billing admin receives usage alerts.

These alerts are not configurable, and are sent when:

- Your organization is approaching a monthly limit of a category of usage (85% of hours used) and when it reaches 100% of the limit, according to your plan.
- Your organization's accumulated average charges for a billing period exceed these thresholds: $200, $450, $700, and $1000. These overage charges are incurred when your organization accumulates more usage than your plan includes for tracked hours, storage, or W&B Weave data ingestion.

For questions about usage or billing, contact your account team or Support.

## Payment methods
This section shows the payment methods on file for your organization. If you have not added a payment method, you will be prompted to do so when you upgrade your plan or add paid storage.

## Billing admin
This section shows the current billing admin. The billing admin is an organization admin, receives all billing-related emails, and can view and manage payment methods.

{{% alert %}}
In W&B Dedicated Cloud, multiple users can be billing admins. In W&B Multi-tenant Cloud, only one user at a time can be the billing admin.
{{% /alert %}}

To change the billing admin or assign the role to additional users:

1. Click **Manage roles**.
1. Search for a user.
1. Click the **Billing admin** field in that user's row.
1. Read the summary, then click **Change billing user**.

## Invoices
If you pay using a credit card, this section allows you to view monthly invoices.
- For Enterprise accounts that pay via wire transfer, this section is blank. For questions, contact your account team.
- If your organization incurs no charges, no invoice is generated.