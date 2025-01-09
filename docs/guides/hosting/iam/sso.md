---
displayed_sidebar: default
title: Configure SSO with OIDC
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Server's support for OpenID Connect (OIDC) compatible identity providers allows for management of user identities and group memberships through external identity providers like Okta, Keycloak, Auth0, Google, and Entra.

## OpenID Connect (OIDC)

W&B Server supports the following OIDC authentication flows for integrating with external Identity Providers (IdPs).
1. Implicit Flow with Form Post 
2. Authorization Code Flow with Proof Key for Code Exchange (PKCE)

These flows authenticate users and provide W&B Server with the necessary identity information (in the form of ID tokens) to manage access control.

The ID token is a JWT that contains the user's identity information, such as their name, username, email, and group memberships. W&B Server uses this token to authenticate the user and map them to appropriate roles or groups in the system.

In the context of W&B Server, the access token is not required. Access tokens are typically used to authorize requests to APIs on behalf of the user, but since W&B Server’s primary concern is user authentication and identity, it only requires the ID token.

You can use environment variables to [configure IAM options](advanced_env_vars.md) for your [Dedicated Cloud](../hosting-options/dedicated_cloud.md) or [Self-managed](../hosting-options/self-managed.md) instance.

To assist with configuring Identity Providers for [Dedicated Cloud](../hosting-options/dedicated_cloud.md) or [Self-managed](../hosting-options/self-managed.md) W&B Server installations, here are some key guidelines to follow for various IdPs. If you’re using the SaaS version of W&B, reach out to [support@wandb.com](mailto:support@wandb.com) for assistance in configuring an Auth0 tenant for your organization.

<Tabs
  defaultValue="cognito"
  values={[
    {label: 'Cognito', value: 'cognito'},
    {label: 'Okta', value: 'okta'},
    {label: 'Entra', value: 'entra'},
  ]}>
  <TabItem value="cognito">

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

![Screenshot of issuer URL in AWS Cognito](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
Do not use the "Cognito domain" for the IDP url. Cognito provides it's discovery document at `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID`
:::

  </TabItem>
  <TabItem value="okta">


1. Login to the Okta Portal at https://login.okta.com/. 

2. On the left side, select **Applications** and then **Applications** again.
![](/images/hosting/okta_select_applications.png)

3. Click on "Create App integration."
![](/images/hosting/okta_create_new_app_integration.png)

4. On the screen named "Create a new app integration," select **OIDC - OpenID Connect** and **Single-Page Application**. Then click "Next."
![](/images/hosting/okta_create_a_new_app_integration.png)

5. On the screen named "New Single-Page App Integration," fill out the values as follows and click **Save**:
    - App integration name, for example "Weights & Biases"
    - Grant type: Check both "Authorization Code" and "Implicit (hybrid)"
    - Sign-in redirect URIs: https://YOUR_W_AND_B_URL/oidc/callback
    - Sign-out redirect URIs: https://YOUR_W_AND_B_URL/logout
    - Assignments: Select **Skip group assignment for now**
![](/images/hosting/okta_new_single_page_app_integration.png)

6. On the overview screen of the Okta application that you just created, make note of the **Client ID** under **Client Credentials** under the **General** tab:
![](/images/hosting/okta_make_note_of_client_id.png)

7. To identify the Okta OIDC Issuer URL, select **Settings** and then **Account** on the left side.
The Okta UI shows the company name under **Organization Contact**.
![](/images/hosting/okta_identify_oidc_issuer_url.png)

The OIDC issuer URL has the following format: https://COMPANY.okta.com. Replace COMPANY with the corresponding value. Make note of it.

  </TabItem>

<TabItem value="entra">
1. Login to the Azure Portal at https://portal.azure.com/.

2. Select "Microsoft Entra ID" service.
![](/images/hosting/entra_select_entra_service.png)

3. On the left side, select "App registrations."
![](/images/hosting/entra_app_registrations.png)

4. On the top, click "New registration."
![](/images/hosting/entra_new_app_registration.png)

    On the screen named "Register an application," fill out the values as follows:
![](/images/hosting/entra_register_an_application.png)

    - Specify a name, for example "Weights and Biases application"
    - By default the selected account type is: "Accounts in this organizational directory only (Default Directory only - Single tenant)." Modify if you need to.
    - Configure Redirect URI as type **Web** with value: `https://YOUR_W_AND_B_URL/oidc/callback`
    - Click "Register."

- Make a note of the "Application (client) ID" and "Directory (tenant) ID." 

![](/images/hosting/entra_app_overview_make_note.png)


5. On the left side, click **Authentication**.
![](/images/hosting/entra_select_authentication.png)

- Under **Front-channel logout URL**, specify: `https://YOUR_W_AND_B_URL/logout`
- Click "Save."

![](/images/hosting/entra_logout_url.png)


6. On the left side, click "Certificates & secrets."
![](/images/hosting/entra_select_certificates_secrets.png)

- Click "Client secrets" and then click "New client secret."
![](/images/hosting/entra_new_secret.png)

    On the screen named "Add a client secret," fill out the values as follows:
![](/images/hosting/entra_add_new_client_secret.png)

  - Enter a description, for example "wandb"
  - Leave "Expires" as is or change if you have to.
  - Click "Add."


- Make a note of the "Value" of the secret. There is no need for the "Secret ID."
![](/images/hosting/entra_make_note_of_secret_value.png)

You should now have made notes of three values:
- OIDC Client ID
- OIDC Client Secret
- Tenant ID is needed for the OIDC Issuer URL

The OIDC issuer URL has the following format: `https://login.microsoftonline.com/${TenantID}/v2.0`

</TabItem>

</Tabs>

## Set up SSO on the W&B Server

To set up SSO, you need administrator privileges and the following information:
- OIDC Client ID
- OIDC Auth method (implicit` or `pkce`)
- OIDC Issuer URL
- OIDC Client Secret (optional; depends on how you have setup your IdP) 

:::info
Should your IdP require a OIDC Client Secret, specify it with the environment variable OIDC_SECRET.
:::

You can configure SSO using either the W&B Server UI or by passing [environment variables](../env-vars.md) to the `wandb/local` pod. The environment variables take precedence over UI.

:::info
If you're unable to log in to your instance after configuring SSO, you can restart the instance with the `LOCAL_RESTORE=true` environment variable set. This outputs a temporary password to the containers logs and disables SSO. Once you've resolved any issues with SSO, you must remove that environment variable to enable SSO again.
:::

<Tabs
  defaultValue="console"
  values={[
    {label: 'System Console', value: 'console'},
    {label: 'System Settings', value: 'settings'},
  ]}>
  <TabItem value="console">

The System Console is the successor to the System Settings page. It is available with the [W&B Kubernetes Operator](../operator.md) based deployment.

1. Refer to [Access the W&B Management Console](../operator.md#access-the-wb-management-console).

2. Navigate to **Settings**, then **Authentication**. Select **OIDC** in the **Type** dropdown.
![](/images/hosting/sso_configure_via_console.png)

3. Enter the values.

4. Click on **Save**.

5. Log out and then log back in, this time using the IdP login screen.

</TabItem>

<TabItem value="settings">

1. Sign in to your Weights&Biases instance. 
2. Navigate to the W&B App. 

![](/images/hosting/system_settings.png)

3. From the dropdown, select **System Settings**:

![](/images/hosting/system_settings_select_settings.png)

4. Enter your Issuer, Client ID, and Authentication Method. 
5. Select **Update settings**.

![](/images/hosting/system_settings_select_update.png)

</TabItem>
</Tabs>

:::info
If you're unable to log in to your instance after configuring SSO, you can restart the instance with the `LOCAL_RESTORE=true` environment variable set. This will output a temporary password to the containers logs and turn off SSO. Once you've resolved any issues with SSO, you must remove that environment variable to enable SSO again.
:::

## Security Assertion Markup Language (SAML)
W&B Server does not support SAML.

