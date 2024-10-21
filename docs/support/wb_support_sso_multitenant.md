---
title: "Does W&B support SSO for Multi-tenant?"
tags:
   - None
---

Yes, W&B supports setting up Single Sign-On (SSO) for the Multi-tenant offering via Auth0. W&B support SSO integration with any OIDC compliant identity provider(ex: Okta, AzureAD etc.). If you have an OIDC provider, please follow the steps below:

* Create a `Single Page Application (SPA)` on your Identity Provider.
* Set `grant_type` to `implicit` flow.
* Set the callback URI to `https://wandb.auth0.com/login/callback`.

**What W&B needs?**

Once you have the above setup, contact your customer success manager(CSM) and let us know the `Client ID` and `Issuer URL` associated with the application.

We'll then set up an Auth0 connection with the above details and enable SSO.