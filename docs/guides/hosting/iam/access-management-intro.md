---
displayed_sidebar: default
---

# Access management


## Instance Admins
The first user to sign up after the W&B Server instance is initially deployed, is automatically assigned the instance `admin` role. The admin can then add additional users to the organization and create teams.

:::note
W&B recommends to have more than one instance admin in an organization. It is a best practice to ensure that admin operations can continue when the primary admin is not available. 
:::

## Enable SSO

W&B strongly recommends and encourages that users authenticate to an organization using Single Sign-On (SSO). To learn more about how to setup SSO with Dedicated cloud or Self-managed instances, refer to [SSO with OIDC](./sso.md) or [SSO with LDAP](./ldap.md).

:::note
`Instance` or `organization` terms are used interchangeably within the context of Dedicated cloud or Self-managed instances.

W&B is actively developing support for multiple organizations in an enterprise instance of Dedicated cloud or Self-managed. If you're interested in utilizing that capability, reach out to your W&B team.
:::


## User auto-provisioning
If Single Sign-On (SSO) is setup for your enterprise W&B Server instance, any user in your company who has access to the instance URL can sign-in to the organization, provided the settings in your SSO provider allow so. When a user signs in for the first time using SSO, their W&B organization user will be automatically created without needing an instance admin to generate a user invite. This is a good alternative for adding users to your W&B organization at scale.

User auto-provisioning with SSO on by default for W&B Server. It is possible to turn it `off` if you would like to selectively add specific users to your W&B organization. If you're on **Dedicated Cloud**, reach out to your W&B team. If you've a **Self-managed** deployment, you can configure the setting `DISABLE_SSO_PROVISIONING=true` for your W&B Server instance.

:::note
If auto-provisioning is on for your W&B Server instance, there may be a way to control which specific users can sign-in to the organization with your SSO provider to restrict the product use to relevant personnel. Extent of that configurability will depend on your SSO provider and is outside the scope of W&B documentation.
:::
