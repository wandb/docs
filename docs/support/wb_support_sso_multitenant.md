---
title: "Does W&B support SSO for Multi-tenant?"
tags:
   - None
---
W&B supports Single Sign-On (SSO) for the Multi-tenant offering through Auth0. SSO integration is compatible with any OIDC-compliant identity provider, such as Okta or Azure AD. To configure an OIDC provider, follow these steps:

* Create a Single Page Application (SPA) on the identity provider.
* Set the `grant_type` to `implicit` flow.
* Set the callback URI to `https://wandb.auth0.com/login/callback`.

**Requirements for W&B**

After completing the setup, contact the customer success manager (CSM) with the `Client ID` and `Issuer URL` for the application. W&B will establish an Auth0 connection using these details and enable SSO.