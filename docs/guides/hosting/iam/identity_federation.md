---
displayed_sidebar: default
title: Manage identity federation with SDK
---

Use identity federation to sign in using your organizational credentials through W&B SDK. If your W&B organization admin has configured SSO for your organization, then you already use your organizational credentials to sign-in to the W&B app UI. In that sense, identity federation is like SSO for W&B SDK, but by using JSON Web Tokens (JWTs) directly. You can use identity federation as an alternative to API keys.

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523) forms the underlying basis for identity federation with SDK.

:::info
Identity federation is available in `Preview` for `Enterprise` plans on all platform types - SaaS Cloud, Dedicated Cloud, and Self-managed instances. Reach out to your W&B team for any questions.
:::

:::note
For the purpose of this document, the terms `identity provider` and `JWT issuer` are used interchangeably. Both refer to one and the same thing in the context of this capability.
:::

## JWT issuer setup

As a first step, an organization admin must set up a federation between your W&B organization and a publicly accessible JWT issuer.

* Go to the **Settings** tab in your organization dashboard
* In the **Authentication** option, press `Set up JWT Issuer`
* Add the JWT issuer URL in the text box and press `Create`

W&B will automatically look for a OIDC discovery document at the path `${ISSUER_URL}/.well-known/oidc-configuration`, and try to find the JSON Web Key Set (JWKS) at a relevant URL in the discovery document. The JWKS is used for real-time validation of the JWTs to ensure that those have been issued by the relevant identity provider.

## Using the JWT to access W&B

Once a JWT issuer has been setup for your W&B organization, users can start accessing the relevant W&B projects using JWTs issued by that identity provider. The mechanism for using JWTs is as follows:

* You must sign-in to the identity provider using one of the mechanisms available in your organization. Some providers can be accessed in an automated manner using an API or SDK, while some can only be accessed using a relevant UI. Reach out to your W&B organization admin or the owner of the JWT issuer for details.
* Once you've retrieved the JWT after signing in to your identity provider, store it in a file at a secure location and configure the absolute file path in an environment variable `WANDB_IDENTITY_TOKEN_FILE`.
* Access your W&B project using the W&B SDK or CLI. The SDK or CLI should automatically detect the JWT and exchange it for a W&B access token after the JWT has been successfully validated. The W&B access token is used to access the relevant APIs for enabling your AI workflows, that is, to log runs, metrics, artifacts and so forth. The access token is by default stored at the path `~/.config/wandb/credentials.json`. You can change that path by specifying the environment variable `WANDB_CREDENTIALS_FILE`.

:::info
JWTs are meant to be short-lived credentials to address the shortcomings of long-lived credentials like API keys, passwords and so forth. Depending on the JWT expiry time configured in your identity provider, you must continuously refresh the JWT and ensure that it's stored in the file referenced by the environment variable `WANDB_IDENTITY_TOKEN_FILE`.

W&B access token also has a default expiry duration, after which the SDK or the CLI automatically try to refresh that using your JWT. If the user JWT has also expired by that time and is not refreshed, that could result in an authentication failure. If possible, the JWT retrieval and post-expiry refresh mechanism should be implemented as part of the AI workload that uses the W&B SDK or CLI.
:::

### JWT validation

As part of the workflow to exchange the JWT for a W&B access token and then access a project, the JWT undergoes following validations:

* The JWT signature is verified using the JWKS at the W&B organization level. This is the first line of defense, and if this fails, that means there's a problem with your JWKS or how your JWT is signed.
* The `iss` claim in the JWT should be equal to the issuer URL configured at the organization level.
* The `sub` claim in the JWT should be equal to the user's email address as configured in the W&B organization.
* The `aud` claim in the JWT should be equal to the name of the W&B organization which houses the project that you are accessing as part of your AI workflow. In case of [Dedicated Cloud](../hosting-options/dedicated_cloud.md) or [Self-managed](../hosting-options/self-managed.md) instances, you could configure an instance-level environment variable `SKIP_AUDIENCE_VALIDATION` to `true` to skip validation of the audience claim, or use `wandb` as the audience.
* The `exp` claim in the JWT is checked to see if the token is valid or has expired and needs to be refreshed.

## External service accounts

W&B has supported built-in service accounts with long-lived API keys for long. With the identity federation capability for SDK and CLI, you can also bring external service accounts that could use JWTs for authentication, though as long as those are issued by the same issuer which is configured at the organization level. A team admin can configure external service accounts within the scope of a team, like the built-in service accounts.

To configure an external service account:

* Go to the **Service Accounts** tab for your team
* Press `New service account`
* Provide a name for the service account, select `Federated Identity` as the `Authentication Method`, provide a `Subject`, and press `Create`

The `sub` claim in the external service account's JWT should be same as what the team admin configures as its subject in the team-level **Service Accounts** tab. That claim is verified as part of [JWT validation](#jwt-validation). The `aud` claim requirement is similar to that for human user JWTs.

When [using an external service account's JWT to access W&B](#using-the-jwt-to-access-wb), it's typically easier to automate the workflow to generate the initial JWT and continuously refresh it. If you would like to attribute the runs logged using an external service account to a human user, you can configure the environment variables `WANDB_USERNAME` or `WANDB_USER_EMAIL` for your AI workflow, similar to how it's done for the built-in service accounts.

:::note
W&B recommends to use a mix of built-in and external service accounts across your AI workloads with different levels of data sensitivity, in order to strike a balance between flexibility and simplicity.
:::


