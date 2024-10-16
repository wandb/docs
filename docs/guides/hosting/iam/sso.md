---
displayed_sidebar: default
title: Configure SSO with OIDC
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Server’s support for OpenID Connect (OIDC) compatible Identity Providers allows for streamlined and secure management of user identities and group memberships through external identity providers like Okta, Keycloak, Auth0, Google, or Microsoft’s Entra. This integration centralizes authentication, simplifying user management and ensuring adherence to security best practices such as single sign-on (SSO) and multi-factor authentication (MFA).

## Open ID Connect

W&B Server supports two OpenID Connect (OIDC) authentication flows for integrating with external Identity Providers (IdPs): 
- Implicit Flow with Form Post and 
- Authorization Code Flow with Proof Key for Code Exchange (PKCE). 

These flows are used to authenticate users and provide W&B Server with the necessary identity information (in the form of ID tokens) to manage access control.

The ID token is a JWT that contains the user’s identity information, including attributes such as name, username, email, and optionally group memberships. This is the primary token W&B Server uses to authenticate the user and map them to appropriate roles or groups in the system.

In the context of W&B Server, the access token is not required. Access tokens are typically used to authorize requests to APIs on behalf of the user, but since W&B Server’s primary concern is user authentication and identity, it only requires the ID token.

In addition to basic [environment variables](../env-vars.md), you can use environment variables to [configure IAM options](advanced_env_vars.md) for your [Dedicated Cloud](../hosting-options/dedicated_cloud.md) or [Self-managed](../hosting-options/self-managed.md) instance.

To assist with configuring Identity Providers for [Dedicated Cloud](../hosting-options/dedicated_cloud.md) or [Self-managed](../hosting-options/self-managed.md) W&B Server installations, here are some key guidelines to follow for various IdPs. If you’re using the SaaS version of W&B, reach out to [support@wandb.com](mailto:support@wandb.com) for assistance in configuring an Auth0 tenant for your organization.

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'Okta', value: 'okta'},
  ]}>
  <TabItem value="aws">

Follow the procedure below to set up AWS Cognito for authorization: 

1. First, sign in to your AWS account and navigate to the [AWS Cognito](https://aws.amazon.com/cognito/) App.

![Because we're only using OIDC for authentication and not authorization, public clients simplify setup](/images/hosting/setup_aws_cognito.png)



2. Provide an allowed callback URL to configure the application in your IdP:
     * Add `http(s)://YOUR-W&B-HOST/oidc/callback` as the callback URL. Replace `YOUR-W&B-HOST` with your W&B host path.

3. If your IdP supports universal logout, set the Logout URL to `http(s)://YOUR-W&B-HOST`. Replace `YOUR-W&B-HOST` with your W&B host path.

For example, if your application was running at `https://wandb.mycompany.com`, you would replace `YOUR-W&B-HOST` with `wandb.mycompany.com`.

The image below demonstrates how to provide allowed callback and sign-out URLs in AWS Cognito.

![If your instance is accessible from multiple hosts, be sure to include all of them here.](/images/hosting/setup_aws_cognito_ui_settings.png)


_wandb/local_ uses the ["implicit" grant with the "form_post" response type](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) by default. 

You can also configure _wandb/local_ to perform an "authorization_code" grant that uses the [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) flow. 

4. Select one or more OAuth grant types to configure how AWS Cognito will deliver tokens to your app.
5. W&B requires specific OpenID Connect (OIDC) scopes. Select the following from AWS Cognito App:
    * "openid" 
    * "profile"
    * "email"

For example, your AWS Cognito App UI should look similar to the following image:

![openid, profile, and email are required](/images/hosting/setup_aws_required_fields.png)

Select the **Auth Method** in the settings page or set the OIDC_AUTH_METHOD environment variable to tell _wandb/local_ which grant to.

:::info
For AWS Cognito providers you must set the Auth Method to "pkce"
:::

6. You need a Client ID and the URL of your OIDC issuer. The OpenID discovery document must be available at `$OIDC_ISSUER/.well-known/openid-configuration` 

For example, with AWS Cognito, you can generate your issuer URL by appending your User Pool ID to the Cognito IdP URL from the **App Integration** tab within the **User Pools** section:

![The issuer URL would be https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
Do not use the "Cognito domain" for the IDP url. Cognito provides it's discovery document at `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID`
:::


<!-- 7. Lastly, provide the OIDC Issuer, Client ID, and Auth method to _wandb/local_ on `https://deploy.wandb.ai/system-admin` or set them as environment variables.

The following image demonstrates how to: enable SSO, provide the OIDC Issuer, Client ID, and the authentication method in the W&B App UI (`https://deploy.wandb.ai/system-admin`): -->

<!-- Once you have everything configured you can provide the Issuer, Client ID, and Auth method to `wandb/local` via `/system-admin` or the environment variables and SSO will be configured.

1. Sign in to your Weights and Biases server 
2. Navigate to the W&B App. 

![](/images/hosting/system_settings.png)

3. From the dropdown, select **System Settings**:

![](/images/hosting/system_settings_select_settings.png)

4. Enter your Issuer, Client ID, and Authentication Method. 
5. Select **Update settings**.

![](/images/hosting/system_settings_select_update.png)

![](/images/hosting/enable_sso.png) -->

  </TabItem>
  <TabItem value="okta">


1. First set up a new application.  Navigate to Okta's App UI and select **Add apps**:

![](/images/hosting/okta.png)

2. Provide a name for App in the **App Integration name** field (for example: Weights and Biases)
3. Select grant type `implicit (hybrid)`

W&B also supports the Authorization Code grant type with PKCE

![](/images/hosting/pkce.png)

4. Provide an allowed callback url:
    * Add the following allowed Callback URL `http(s)://YOUR-W&B-HOST/oidc/callback`.

5. If your IdP supports universal logout, set the **Logout URL** to `http(s)://YOUR-W&B-HOST`.

![](/images/hosting/redirect_uri.png)
For example, if your application runs in a local host on port 8080 (`https://localhost:8080`),
the redirect URI would look like: `https://localhost:8080/oidc/callback`.

6. Set the sign-out redirect to `http(s)://YOUR-W&B-HOST/logout` in the **Sign-out redirects URIs** field: 

![](/images/hosting/signout_redirect.png)

7. Provide the OIDC Issuer, Client ID, and Auth method to wandb/local on https://deploy.wandb.ai/system-admin or set them as environment variables.

  </TabItem>
</Tabs>

## Configure SSO on the W&B App

Once you have everything configured you can provide the Issuer, Client ID, and Auth method to `wandb/local` on the W&B App or set environment variables. The following procedure walks you through the steps to configure SSO with the W&B App UI:

1. Sign in to your Weights and Biases server 
2. Navigate to the W&B App. 

![](/images/hosting/system_settings.png)

3. From the dropdown, select **System Settings**:

![](/images/hosting/system_settings_select_settings.png)

4. Enter your Issuer, Client ID, and Authentication Method. 
5. Select **Update settings**.

![](/images/hosting/system_settings_select_update.png)

:::info
If you're unable to log in to your instance after configuring SSO, you can restart the instance with the `LOCAL_RESTORE=true` environment variable set. This will output a temporary password to the containers logs and disable SSO. Once you've resolved any issues with SSO, you must remove that environment variable to enable SSO again.
:::

## Security Assertion Markup Language (SAML)
W&B Server does not support SAML.

