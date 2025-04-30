---
title: あなたの組織を管理する
menu:
  default:
    identifier: ja-guides-hosting-iam-access-management-manage-organization
    parent: access-management
weight: 1
---

As an admin of an organization you can [manage individual users]({{< relref path="#add-and-manage-users" lang="ja" >}}) within your organization and [manage teams]({{< relref path="#add-and-manage-teams" lang="ja" >}}).

組織の管理者として、組織内の[個々のユーザーを管理]({{< relref path="#add-and-manage-users" lang="ja" >}})し、[チームを管理]({{< relref path="#add-and-manage-teams" lang="ja" >}})できます。 

As a team admin you can [manage teams]({{< relref path="#add-and-manage-teams" lang="ja" >}}).

チーム管理者として、[チームを管理]({{< relref path="#add-and-manage-teams" lang="ja" >}})できます。

{{% alert %}}
The following workflow applies to users with instance admin roles. Reach out to an admin in your organization if you believe you should have instance admin permissions.
{{% /alert %}}

以下のワークフローは、インスタンス管理者の役割を持つユーザーに適用されます。インスタンス管理者の権限が必要だと思われる場合は、組織の管理者に問い合わせてください。

If you are looking to simplify user management in your organization, refer to [Automate user and team management]({{< relref path="../automate_iam.md" lang="ja" >}}).

組織内のユーザー管理を簡素化したい場合は、[ユーザーとチームの管理を自動化する]({{< relref path="../automate_iam.md" lang="ja" >}})を参照してください。


## Change the name of your organization
{{% alert %}}
The following workflow only applies to W&B Multi-tenant SaaS Cloud.
{{% /alert %}}

## 組織名の変更
{{% alert %}}
以下のワークフローはW&BマルチテナントSaaSクラウドにのみ適用されます。
{{% /alert %}}

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Settings**.
3. Within the **Settings** tab, select **General**.
4. Select the **Change name** button.
5. Within the modal that appears, provide a new name for your organization and select the **Save name** button.

1. https://wandb.ai/home に移動します。
2. ページの右上隅にある **ユーザーメニュー** のドロップダウンを選択します。ドロップダウン内の **アカウント** セクションで **設定** を選択します。
3. **設定** タブ内で **一般** を選択します。
4. **名前を変更** ボタンを選択します。
5. 表示されるモーダル内に、新しい組織名を入力し、**名前を保存** ボタンを選択します。

## Add and manage users

As an admin, use your organization's dashboard to:
- Invite or remove users.
- Assign or update a user's organization role, and create custom roles.
- Assign the billing admin.

管理者として、組織のダッシュボードを使用して次のことを行います。
- ユーザーを招待または削除する。
- ユーザーの組織の役割を割り当てまたは更新し、カスタム役割を作成する。
- 請求管理者を割り当てる。

There are several ways an organization admin can add users to an organization:

組織の管理者がユーザーを組織に追加する方法はいくつかあります。

1. Member-by-invite
2. Auto provisioning with SSO
3. Domain capture

1. 招待によるメンバー
2. SSOによる自動プロビジョニング
3. ドメインキャプチャ

### Seats and pricing

The proceeding table summarizes how seats work for Models and Weave:

### シートと価格

以下の表は、Models と Weave のシートの仕組みをまとめたものです。

| Product |Seats | Cost based on |
| ----- | ----- | ----- |
| Models | Pay per set | How many Models paid seats you have, and how much usage you’ve accrued determines your overall subscription cost. Each user can be assigned one of the three available seat types: Full, Viewer, and No-Access |
| Weave | Free  | Usage based |

| 製品 | シート | 費用基準 |
| ----- | ----- | ----- |
| Models | セットごとの支払い | 支払い済みの Models シートの数と、累積した使用量が総合的なサブスクリプション費用を決定します。各ユーザーには、3 つの利用可能なシートタイプのいずれかを割り当てることができます: Full, Viewer, No-Access |
| Weave | 無料 | 使用量に基づく |

### Invite a user

admins can invite users to their organization, as well as specific teams within the organization.

### ユーザーを招待する

管理者は、組織内の特定のチームに加えてユーザーを組織に招待できます。

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
1. Navigate to https://wandb.ai/home.
1. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Users**.
3. Select **Invite new user**.
4. In the modal that appears, provide the email or username of the user in the **Email or username** field.
5. (Recommended) Add the user to a team from the **Choose teams** dropdown menu.
6. From the **Select role** dropdown, select the role to assign to the user. You can change the user's role at a later time. See the table listed in [Assign a role]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) for more information about possible roles.
7. Choose the **Send invite** button.

1. https://wandb.ai/home に移動します。
1. ページの右上隅にある **ユーザーメニュー** のドロップダウンを選択します。ドロップダウン内の **アカウント** セクションで **ユーザー** を選択します。
3. **新しいユーザーを招待する** を選択します。
4. 表示されるモーダルで、**メールまたはユーザー名** フィールドにそのユーザーのメールまたはユーザー名を入力します。
5. (推奨) **チームを選択** ドロップダウンメニューから、そのユーザーをチームに追加します。
6. **役割を選択** ドロップダウンから、そのユーザーに割り当てる役割を選択します。ユーザーの役割は後で変更可能です。[役割を割り当てる]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) に記載されている表を参照して、可能な役割について更に詳しく知ってください。
7. **招待を送信** ボタンを選択します。

W&B sends an invite link using a third-party email server to the user's email after you select the **Send invite** button. A user can access your organization once they accept the invite.
{{% /tab %}}

W&Bはサードパーティのメールサーバーを使って、**招待を送信** ボタンを選択した後にユーザーのメールに招待リンクを送信します。ユーザーが招待を受け入れると、あなたの組織にアクセスできるようになります。
{{% /tab %}}

{{% tab header="Dedicated or Self-managed" value="dedicated"%}}
1. Navigate to `https://<org-name>.io/console/settings/`. Replace `<org-name>` with your organization name.
2. Select the **Add user** button
3. Within the modal that appears, provide the email of the new user in the **Email** field.
4. Select a role to assign to the user from the **Role** dropdown. You can change the user's role at a later time. See the table listed in [Assign a role]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) for more information about possible roles.
5. Check the **Send invite email to user** box if you want W&B to send an invite link using a third-party email server to the user's email.
6. Select the **Add new user** button.
{{% /tab %}}
{{< /tabpane >}}

1. `https://<org-name>.io/console/settings/` に移動します。`<org-name>` はあなたの組織の名前に置き換えてください。
2. **ユーザーを追加** ボタンを選択します。
3. 表示されるモーダルで、新しいユーザーのメールアドレスを **メール** フィールドに入力します。
4. **役割** ドロップダウンからユーザーに割り当てる役割を選択します。ユーザーの役割は後で変更可能です。[役割を割り当てる]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) に記載されている表を参照して、可能な役割について更に詳しく知ってください。
5. ユーザーのメールにサードパーティのメールサーバーを使って招待リンクを送信したい場合は、**招待メールをユーザーに送信** ボックスにチェックを入れます。
6. **新しいユーザーを追加** ボタンを選択します。
{{% /tab %}}
{{< /tabpane >}}

### Auto provision users

A W&B user with matching email domain can sign in to your W&B Organization with Single Sign-On (SSO) if you configure SSO and your SSO provider permits it. SSO is available for all Enterprise licenses.

### ユーザーの自動プロビジョニング

一致するメールドメインを持つW&Bユーザーは、SSOを設定し、SSOプロバイダーが許可した場合、SSOを使ってW&B組織にサインインすることができます。SSOはすべてのエンタープライズライセンスに利用可能です。

{{% alert title="Enable SSO for authentication" %}}
W&B strongly recommends and encourages that users authenticate using Single Sign-On (SSO). Reach out to your W&B team to enable SSO for your organization. 

W&Bは、ユーザーがSSOを使って認証することを強く推奨しています。組織のSSOを有効にするためには、W&Bチームに連絡してください。

To learn more about how to setup SSO with Dedicated cloud or Self-managed instances, refer to [SSO with OIDC]({{< relref path="../authentication/sso.md" lang="ja" >}}) or [SSO with LDAP]({{< relref path="../authentication/ldap.md" lang="ja" >}}).{{% /alert %}}

Dedicated cloudまたはSelf-managedインスタンスでSSOを設定する方法についての詳細は、[SSO with OIDC]({{< relref path="../authentication/sso.md" lang="ja" >}})または[SSO with LDAP]({{< relref path="../authentication/ldap.md" lang="ja" >}})を参照してください。{{% /alert %}}

W&B assigned auto-provisioning users "Member" roles by default. You can change the role of auto-provisioned users at any time.

W&Bはデフォルトで自動プロビジョニングされたユーザーに「メンバー」役割を割り当てます。自動プロビジョニングされたユーザーの役割はいつでも変更可能です。

Auto-provisioning users with SSO is on by default for Dedicated cloud instances and Self-managed deployments. You can turn off auto provisioning. Turning auto provisioning off enables you to selectively add specific users to your W&B organization.

Dedicated cloudインスタンスおよびSelf-managedデプロイメントでは、SSOを使ったユーザーの自動プロビジョニングがデフォルトでオンになっています。自動プロビジョニングをオフにすることができ、特定のユーザーを選んでW&B組織に追加することができます。

The proceeding tabs describe how to turn off SSO based on deployment type:

以下のタブでは、デプロイメントタイプに基づいてSSOをオフにする方法を説明します：

{{< tabpane text=true >}}
{{% tab header="Dedicated cloud" value="dedicated" %}}
Reach out to your W&B team if you are on Dedicated cloud instance and you want to turn off auto provisioning with SSO.
{{% /tab %}}

Dedicated cloudインスタンスを使用している場合で、自動プロビジョニングをオフにしたい場合は、W&Bチームに連絡してください。
{{% /tab %}}

{{% tab header="Self-managed" value="self_managed" %}}
Use the W&B Console to turn off auto provisioning with SSO:

{{% tab header="Self-managed" value="self_managed" %}}
W&Bコンソールを使用してSSOを使った自動プロビジョニングをオフにします。

1. Navigate to `https://<org-name>.io/console/settings/`. Replace `<org-name>` with your organization name.
2. Choose **Security** 
3. Select the **Disable SSO Provisioning** to turn off auto provisioning with SSO.

1. `https://<org-name>.io/console/settings/` に移動します。`<org-name>` をあなたの組織名に置き換えてください。
2. **セキュリティ** を選択します。
3. **SSOプロビジョニングを無効にする** を選択して、SSOを使った自動プロビジョニングをオフにします。

{{% /tab %}}
{{< /tabpane >}}

{{% alert title="" %}}
Auto provisioning with SSO is useful for adding users to an organization at scale because organization admins do not need to generate individual user invitations.
{{% /alert %}}

SSOを使った自動プロビジョニングは、大規模にユーザーを組織に追加するのに役立ちます。なぜなら、組織の管理者が個別のユーザー招待を生成する必要がないからです。
{{% /alert %}}

### Create custom roles
{{% alert %}}
An Enterprise license is required to create or assign custom roles on Dedicated cloud or Self-managed deployments.
{{% /alert %}}

### カスタム役割を作成する
{{% alert %}}
Dedicated cloudまたはSelf-managedデプロイメントでカスタム役割を作成または割り当てるには、エンタープライズライセンスが必要です。
{{% /alert %}}

Organization admins can compose a new role based on either the View-Only or Member role and add additional permissions to achieve fine-grained access control. Team admins can assign a custom role to a team member. Custom roles are created at the organization level but are assigned at the team level.

組織の管理者は、View-Only または Member 役割に基づいて新しい役割を作成し、追加の許可を追加することで詳細なアクセス制御を実現できます。チーム管理者は、チームメンバーにカスタム役割を割り当てることができます。カスタム役割は組織レベルで作成され、チームレベルで割り当てられます。

To create a custom role:

カスタム役割を作成するには：

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
1. Navigate to https://wandb.ai/home.
1. In the upper right corner of the page, select the **User menu** dropdown. Within the **Account** section of the dropdown, select **Settings**.
1. Click **Roles**.
1. In the **Custom roles** section, click **Create a role**.
1. Provide a name for the role. Optionally provide a description.
1. Choose the role to base the custom role on, either **Viewer** or **Member**.
1. To add permissions, click the **Search permissions** field, then select one or more permissions to add.
1. Review the **Custom role permissions** section, which summarizes the permissions the role has.
1. Click **Create Role**.

1. https://wandb.ai/home に移動します。
1. ページの右上隅にある **ユーザーメニュー** のドロップダウンを選択します。ドロップダウン内の **アカウント** セクションで **設定** を選択します。
1. **役割** をクリックします。
1. **カスタム役割** セクションで **役割を作成** をクリックします。
1. 役割の名前を入力します。必要に応じて説明を追加します。
1. カスタム役割のベースとする役割を **Viewer** または **Member** から選択します。
1. 許可を追加するには、**許可を検索** フィールドをクリックし、追加する1つ以上の許可を選択します。
1. 役割が持つ許可を要約している **カスタム役割の許可** セクションを確認します。
1. **役割を作成** をクリックします。
{{% /tab %}}

{{% tab header="Dedicated or Self-managed" value="dedicated"%}}
Use the W&B Console to create custom roles:

{{% tab header="Dedicated or Self-managed" value="dedicated" %}}
W&Bコンソールを使ってカスタム役割を作成します：

1. Navigate to `https://<org-name>.io/console/settings/`. Replace `<org-name>` with your organization name.
1. In the **Custom roles** section, click **Create a role**.
1. Provide a name for the role. Optionally provide a description.
1. Choose the role to base the custom role on, either **Viewer** or **Member**.
1. To add permissions, click the **Search permissions** field, then select one or more permissions to add.
1. Review the **Custom role permissions** section, which summarizes the permissions the role has.
1. Click **Create Role**.

1. `https://<org-name>.io/console/settings/` に移動します。`<org-name>` をあなたの組織名に置き換えてください。
1. **カスタム役割** セクションで **役割を作成** をクリックします。
1. 役割の名前を入力します。必要に応じて説明を追加します。
1. カスタム役割のベースとする役割を **Viewer** または **Member** から選択します。
1. 許可を追加するには、**許可を検索** フィールドをクリックし、追加する1つ以上の許可を選択します。
1. 役割が持つ許可を要約している **カスタム役割の許可** セクションを確認します。
1. **役割を作成** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

A team admin can now assign the custom role to members of a team from the [Team settings]({{< relref path="#invite-users-to-a-team" lang="ja" >}}).

チーム管理者は、今後、[チーム設定]({{< relref path="#invite-users-to-a-team" lang="ja" >}}) からチームのメンバーにカスタム役割を割り当てることができます。

### Domain capture
Domain capture helps your employees join your company's organization to ensure new users do not create assets outside of your company jurisdiction.

### ドメインキャプチャ
ドメインキャプチャは、従業員があなたの会社の組織に参加する手助けをし、新しいユーザーが会社の管轄外で資産を作成することがないようにします。

{{% alert title="Domains must be unique" %}}
Domains are unique identifiers. This means that you can not use a domain that is already in use by another organization.
{{% /alert %}}

{{% alert title="ドメインはユニークである必要があります" %}}
ドメインは一意の識別子です。つまり、他の組織ですでに使用されているドメインは使用できません。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="Multi-tenant SaaS Cloud" value="saas" %}}
Domain capture lets you automatically add people with a company email address, such as `@example.com`, to your W&B SaaS cloud organization. This helps all your employees join the right organization and ensures that new users do not create assets outside of your company jurisdiction.

ドメインキャプチャを使用すると、`@example.com` のような会社のメールアドレスを持つ人々を自動的にW&B SaaSクラウド組織に追加できます。これにより、すべての従業員が適切な組織に参加し、新しいユーザーが会社の管轄外で資産を作成することを防ぎます。

This table summarizes the behavior of new and existing users with and without domain capture enabled:

この表は、ドメインキャプチャが有効か無効かによる、新しいユーザーと既存のユーザーの振る舞いを要約したものです。

| | With domain capture | Without domain capture |
| ----- | ----- | ----- |
| New users | Users who sign up for W&B from verified domains are automatically added as members to your organization’s default team. They can choose additional teams to join at sign up, if you enable team joining. They can still join other organizations and teams with an invitation. | Users can create W&B accounts without knowing there is a centralized organization available. |
| Invited users | Invited users automatically join your organization when accepting your invite. Invited users are not automatically added as members to your organization’s default team. They can still join other organizations and teams with an invitation. | Invited users automatically join your organization when accepting your invite. They can still join other organizations and teams with an invitation.|
| Existing users | Existing users with verified email addresses from your domains can join your organization’s teams within the W&B App. All data that existing users create before joining your organization remains. W&B does not migrate the existing user's data. | Existing W&B users may be spread across multiple organizations and teams.|

| | ドメインキャプチャあり | ドメインキャプチャなし |
| ----- | ----- | ----- |
| 新しいユーザー | 確認済みドメインからW&Bに登録したユーザーは自動的に組織のデフォルトチームにメンバーとして追加されます。チーム参加を有効にしている場合、登録時に追加のチームを選択することができます。招待を受け入れて他の組織やチームにも参加することができます。 | ユーザーは、集中化された組織があることを知らずにW&Bアカウントを作成することができます。 |
| 招待されたユーザー | 招待を受け入れたときに招待されたユーザーは自動的にあなたの組織に参加します。招待されたユーザーは組織のデフォルトチームに自動的にメンバーとして追加されません。招待を受けて他の組織やチームにも参加できます。 | 招待を受け入れたときに招待されたユーザーは自動的にあなたの組織に参加します。招待を受けて他の組織やチームにも参加できます。| 
| 既存のユーザー | あなたのドメインからの確認済みメールアドレスを持つ既存のユーザーは、W&Bアプリ内の組織のチームに参加できます。組織に参加する前に既存のユーザーが作成したすべてのデータは残ります。W&Bは既存ユーザーのデータを移行しません。 | 既存のW&Bユーザーは、複数の組織やチームに分散している可能性があります。|

To automatically assign non-invited new users to a default team when they join your organization:

招待されていない新しいユーザーを組織に参加した際にデフォルトチームに自動的に割り当てるには：

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Settings**.
3. Within the **Settings** tab, select **General**.
4. Choose the **Claim domain** button within **Domain capture**.
5. Select the team that you want new users to automatically join from the **Default team** dropdown. If no teams are available, you'll need to update team settings. See the instructions in [Add and manage teams]({{< relref path="#add-and-manage-teams" lang="ja" >}}).
6. Click the **Claim email domain** button.

1. https://wandb.ai/home に移動します。
2. ページの右上隅にある **ユーザーメニュー** のドロップダウンを選択します。ドロップダウンから **設定** を選択します。
3. **設定** タブ内で **一般** を選択します。
4. **ドメインキャプチャ** 内の **ドメインを請求する** ボタンを選択します。
5. **デフォルトチーム** ドロップダウンから新しいユーザーが自動的に参加するチームを選択します。利用可能なチームがない場合は、チーム設定を更新する必要があります。[チームを追加し管理する]({{< relref path="#add-and-manage-teams" lang="ja" >}})の指示を参照してください。
6. **メールドメインを請求する** ボタンをクリックします。

You must enable domain matching within a team's settings before you can automatically assign non-invited new users to that team.

招待されていない新しいユーザーを自動的にそのチームに割り当てる前に、チームの設定でドメインマッチングを有効にする必要があります。

1. Navigate to the team's dashboard at `https://wandb.ai/<team-name>`. Where `<team-name>` is the name of the team you want to enable domain matching.
2. Select **Team settings** in the global navigation on the left side of the team's dashboard.
3. Within the **Privacy** section, toggle the "Recommend new users with matching email domains join this team upon signing up" option.

1. `https://wandb.ai/<team-name>` にあるチームのダッシュボードに移動します。ここで `<team-name>` はドメインマッチングを有効にしたいチームの名前です。
2. チームのダッシュボードの左側のグローバルナビゲーションで **チーム設定** を選択します。
3. **プライバシー** セクションで、「一致するメールドメインを持つ新しいユーザーに、サインアップ時にこのチームに参加することを推奨する」オプションを切り替えます。

{{% /tab %}}
{{% tab header="Dedicated or Self-managed" value="dedicated" %}}
Reach out to your W&B Account Team if you use Dedicated or Self-managed deployment type to configure domain capture. Once configured, your W&B SaaS instance automatically prompts users who create a W&B account with your company email address to contact your admin to request access to your Dedicated or Self-managed instance.

専用または自己管理のデプロイメントタイプを使用してドメインキャプチャを構成する際は、W&Bアカウントチームにご連絡ください。設定が完了すると、会社のメールアドレスでW&Bアカウントを作成したユーザーに対し、専用または自己管理のインスタンスへのアクセスを求めるために管理者に連絡するよう自動的に促されます。

| | With domain capture | Without domain capture |
| ----- | ----- | -----|
| New users | Users who sign up for W&B on SaaS cloud from verified domains are automatically prompted to contact an admin with an email address you customize. They can still create an organizations on SaaS cloud to trial the product. | Users can create W&B SaaS cloud accounts without learning their company has a centralized dedicated instance. | 
| Existing users | Existing W&B users may be spread across multiple organizations and teams.| Existing W&B users may be spread across multiple organizations and teams.|

| | ドメインキャプチャあり | ドメインキャプチャなし |
| ----- | ----- | -----|
| 新しいユーザー | 確認済みドメインからSaaSクラウド上のW&Bにサインアップしたユーザーは、カスタマイズしたメールアドレスを持つ管理者に連絡するように自動的に促されます。プロダクトのトライアルのためにSaaSクラウド上に新しい組織を作成することもできます。 | ユーザーは、会社に集中化された専用インスタンスがあることを知らないまま、W&B SaaSクラウドアカウントを作成することができます。 | 
| 既存のユーザー | 既存のW&Bユーザーは、複数の組織やチームに分散している可能性があります。| 既存のW&Bユーザーは、複数の組織やチームに分散している可能性があります。|

{{% /tab %}}
{{< /tabpane >}}


### Assign or update a user's role

Every member in an Organization has an organization role and seat for both W&B Models and Weave. The type of seat they have determines both their billing status and the actions they can take in each product line.

### ユーザーの役割を割り当てまたは更新する

組織内のすべてのメンバーは、W&B Models および Weave のための組織役割とシートを持っています。彼らのシートタイプによって、彼らの請求状況と各製品ラインで取ることのできるアクションが決まります。

You initially assign an organization role to a user when you invite them to your organization. You can change any user's role at a later time.

組織に招待する際に、ユーザーに組織の役割を初めて割り当てます。後でどのユーザーの役割も変更できます。

A user within an organization can have one of the proceeding roles:

組織内のユーザーは、以下のいずれかの役割を持つことができます：

| Role | Descriptions |
| ----- | ----- |
| admin| A instance admin who can add or remove other users to the organization, change user roles, manage custom roles, add teams and more. W&B recommends ensuring there is more than one admin in the event that your admin is unavailable. |
| Member | A regular user of the organization, invited by an instance admin. A organization member cannot invite other users or manage existing users in the organization. |
| Viewer (Enterprise-only feature) | A view-only user of your organization, invited by an instance admin. A viewer only has read access to the organization and the underlying teams that they are a member of. |
|Custom Roles (Enterprise-only feature) | Custom roles allow organization admins to compose new roles by inheriting from the preceding View-Only or Member roles, and adding additional permissions to achieve fine-grained access control. Team admins can then assign any of those custom roles to users in their respective teams.|

| 役割 | 説明 |
| ----- | ----- |
| 管理者 | 他のユーザーを組織に追加または削除し、ユーザーの役割を変更し、カスタム役割を管理し、チームを追加することができるインスタンス管理者。管理者が不在の場合に備えて、W&Bは複数の管理者がいることを推奨しています。 |
| メンバー | インスタンス管理者によって招待された組織の通常のユーザー。組織メンバーは、他のユーザーを招待したり、組織内の既存のユーザーを管理したりすることはできません。 |
| ビューアー (エンタープライズのみの機能) | インスタンス管理者によって招待された、組織のビュー専用ユーザー。ビューアーは、組織と彼らがメンバーである基盤となるチームに対して読み取り専用アクセスを持ちます。 |
| カスタムロール (エンタープライズのみの機能) | カスタム役割は、組織の管理者が前述の View-Only または Member 役割を継承し、追加の許可を追加して微細かいアクセス制御を達成するために作成することができます。チーム管理者は、その役割をそれぞれのチーム内のユーザーに割り当てることができます。|

To change a user's role:

ユーザーの役割を変更するには：

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
4. Provide the name or email of the user in the search bar.
5. Select a role from the **TEAM ROLE** dropdown next to the name of the user.

1. https://wandb.ai/home に移動します。
2. ページの右上隅にある **ユーザーメニュー** ドロップダウンを選択します。ドロップダウンから **ユーザー** を選択します。
4. 検索バーにユーザーの名前またはメールアドレスを入力します。
5. **チーム役割** ドロップダウンからユーザー名の横にある役割を選択します。

### Assign or update a user's access

A user within an organization has one of the proceeding model seat or weave access types: full, viewer, or no access.

### ユーザーのアクセスを割り当てまたは更新する

組織内のユーザーは、以下のいずれかのモデルシートまたは Weave アクセスタイプを持っています：フル、ビューアー、またはアクセスなし。

| Seat type | Description |
| ----- | ----- |
| Full | Users with this role type have full permissions to write, read, and export data for Models or Weave. |
| Viewer | A view-only user of your organization. A viewer only has read access to the organization and the underlying teams that they are a part of, and view only access to Models or Weave. |
| No access | Users with this role have no access to the Models or Weave products. |

| シートタイプ | 説明 |
| ----- | ----- |
| フル | この役割タイプのユーザーは、Models または Weave のデータを書き込み、読み取り、およびエクスポートするための完全な権限を持ちます。 |
| ビューアー | あなたの組織のビュー専用ユーザー。ビューアーは、組織とその基となるチームに対してのみ読み取りアクセスを持ち、Models または Weave に対してビュー専用アクセスを持ちます。 |
| アクセスなし | この役割を持つユーザーは、Models または Weave 製品へのアクセスはありません。|

Model seat type and weave access type are defined at the organization level, and inherited by the team. If you want to change a user's seat type, navigate to the organization settings and follow the proceeding steps:

モデルシートタイプと Weave アクセスタイプは組織レベルで定義され、チームに継承されます。ユーザーのシートタイプを変更したい場合は、組織の設定に移動し、次のステップに従ってください：

1. For SaaS users, navigate to your organization's settings at `https://wandb.ai/account-settings/<organization>/settings`. Ensure to replace the values enclosed in angle brackets (`<>`) with your organization name. For other Dedicated and Self-managed deployments, navigate to `https://<your-instance>.wandb.io/org/dashboard`.
2. Select the **Users** tab.
3. From the **Role** dropdown, select the seat type you want to assign to the user.

1. SaaS ユーザーの場合、`https://wandb.ai/account-settings/<organization>/settings` にある組織の設定に移動します。角括弧（`<>`）で囲まれた値を組織名に置き換えてください。他の専用または自己管理のデプロイメントの場合は、`https://<your-instance>.wandb.io/org/dashboard` に移動します。
2. **ユーザー** タブを選択します。
3. **役割** ドロップダウンからユーザーに割り当てたいシートタイプを選択します。

{{% alert %}}
The organization role and subscription type determines which seat types are available within your organization.
{{% /alert %}}

{{% alert %}}
組織の役割とサブスクリプションタイプが、あなたの組織内で利用可能なシートタイプを決定します。
{{% /alert %}}

### Remove a user

### ユーザーを削除する

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
3. Provide the name or email of the user in the search bar.
4. Select the ellipses or three dots icon (**...**) when it appears.
5. From the dropdown, choose **Remove member**.

1. https://wandb.ai/home に移動します。
2. ページの右上隅にある **ユーザーメニュー** のドロップダウンを選択します。ドロップダウンから **ユーザー** を選択します。
3. 検索バーにユーザーの名前またはメールアドレスを提供します。
4. 表示されたら、三点リーダーまたは3つの点のアイコン（**...**）を選択します。
5. ドロップダウンから **メンバーを削除** を選択します。

### Assign the billing admin

### 請求管理者を割り当てる

1. Navigate to https://wandb.ai/home.
2. In the upper right corner of the page, select the **User menu** dropdown. From the dropdown, choose **Users**.
3. Provide the name or email of the user in the search bar.
4. Under the **Billing admin** column, choose the user you want to assign as the billing admin.

1. https://wandb.ai/home に移動します。
2. ページの右上隅にある **ユーザーメニュー** のドロップダウンを選択します。ドロップダウンから **ユーザー** を選択します。
3. 検索バーにユーザーの名前またはメールアドレスを入力します。
4. **請求管理者** 列の下で、請求管理者として割り当てたいユーザーを選択します。


## Add and manage teams

## チームを追加し管理する

Use your organization's dashboard to create and manage teams within your organization. An organization admin or a team admin can:

組織のダッシュボードを使って、組織内でチームを作成し管理します。組織の管理者またはチーム管理者は、以下のことができます：

- Invite users to a team or remove users from a team.
- Manage a team member's roles.
- Automate the addition of users to a team when they join your organization.
- Manage team storage with the team's dashboard at `https://wandb.ai/<team-name>`.

- ユーザーをチームに招待したり、チームからユーザーを削除したりする。
- チームメンバーの役割を管理する。
- 組織に参加した時にユーザーをチームに自動的に追加する。
- `https://wandb.ai/<team-name>` にあるチームのダッシュボードでチームストレージを管理する。

### Create a team

### チームを作成する

Use your organization's dashboard to create a team:

組織のダッシュボードを使用してチームを作成します：

1. Navigate to https://wandb.ai/home.
2. Select **Create a team to collaborate** on the left navigation panel underneath **Teams**.
{{< img src="/images/hosting/create_new_team.png" alt="" >}}
3. Provide a name for your team in the **Team name** field in the modal that appears.
4. Choose a storage type.
5. Select the **Create team** button.

1. https://wandb.ai/home に移動します。
2. **チーム** の下の左側のナビゲーションパネルで **コラボレーション用のチームを作成** を選択します。
{{< img src="/images/hosting/create_new_team.png" alt="" >}}
3. 表示されるモーダルで **チーム名** フィールドにチームの名前を入力します。
4. ストレージタイプを選択します。
5. **チームを作成** ボタンを選択します。

After you select **Create team** button, W&B redirects you to a new team page at `https://wandb.ai/<team-name>`. Where `<team-name>` consists of the name you provide when you create a team.

**チームを作成** ボタンを選択すると、W&Bは `https://wandb.ai/<team-name>` の新しいチームページにリダイレクトします。`<team-name>` はチーム作成時に入力した名前を使用します。

Once you have a team, you can add users to that team.

チームを持ったら、そのチームにユーザーを追加することができます。

### Invite users to a team

### チームにユーザーを招待する

Invite users to a team in your organization. Use the team's dashboard to invite users using their email address or W&B username if they already have a W&B account.

組織内のチームにユーザーを招待します。ユーザーがすでにW&Bアカウントを持っている場合は、メールアドレスまたはW&Bユーザー名を使用してチームのダッシュボードからユーザーを招待します。

1. Navigate to `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the global navigation on the left side of the dashboard.
{{< img src="/images/hosting/team_settings.png" alt="" >}}
3. Select the **Users** tab.
4. Choose on **Invite a new user**.
5. Within the modal that appears, provide the email of the user in the **Email or username** field and select the role to assign to that user from the **Select a team** role dropdown. For more information about roles a user can have in a team, see [Team roles]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}).
6. Choose on the **Send invite** button.

1. `https://wandb.ai/<team-name>` に移動します。
2. ダッシュボードの左側のグローバルナビゲーションで **チーム設定** を選択します。
{{< img src="/images/hosting/team_settings.png" alt="" >}}
3. **ユーザー** タブを選択します。
4. **新しいユーザーを招待する** を選びます。
5. 表示されるモーダルで、**メールまたはユーザー名** フィールドにそのユーザーのメールを提供し、**チームを選択する** ドロップダウンからそのユーザーに割り当てる役割を選択します。チーム内でユーザーが持つことができる役割について詳しくは、[チーム役割]({{< relref path="#assign-or-update-a-team-members-role" lang="ja" >}}) を参照してください。
6. **招待を送信** ボタンを選びます。

By default, only a team or instance admin can invite members to a team. To change this behavior, refer to [Team settings]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ja" >}}).

デフォルトでは、チームまたはインスタンスの管理者のみがメンバーをチームに招待できます。この動作を変更するには、[チーム設定]({{< relref path="/guides/models/app/settings-page/team-settings.md#privacy" lang="ja" >}})を参照してください。

In addition to inviting users manually with email invites, you can automatically add new users to a team if the new user's [email matches the domain of your organization]({{< relref path="#domain-capture" lang="ja" >}}).

メール招待でユーザーを手動で招待することに加えて、新しいユーザーの[メールがあなたの組織のドメインと一致する]({{< relref path="#domain-capture" lang="ja" >}})場合は、自動的に新しいユーザーをチームに追加できます。

### Match members to a team organization during sign up

### 新メンバーを登録時にチーム組織と一致させる

Allow new users within your organization discover Teams within your organization when they sign-up. New users must have a verified email domain that matches your organization's verified email domain. Verified new users can view a list of verified teams that belong to an organization when they sign up for a W&B account.

新規ユーザーがサインアップ時に組織内のチームを発見できるようにします。新しいユーザーは、あなたの組織の確認済みメールドメインと一致する確認済みのメールドメインを持っている必要があります。確認済みの新規ユーザーは、W&Bアカウントに登録する際に組織に属する確認済みのチームの一覧を見ることができます。

An organization admin must enable domain claiming. To enable domain capture, see the steps described in [Domain capture]({{< relref path="#domain-capture" lang="ja" >}}).

組織の管理者はドメイン請求を有効にする必要があります。ドメインキャプチャを有効にするには、[ドメインキャプチャ]({{< relref path="#domain-capture" lang="ja" >}})に記載されている手順を参照してください。

### Assign or update a team member's role

### チームメンバーの役割を割り当てまたは更新する

1. Select the account type icon next to the name of the team member.
2. From the drop-down, choose the account type you want that team member to possess.

1. チームメンバーの名前の横にあるアカウントタイプのアイコンを選択します。
2. ドロップダウンから、そのチームメンバーが持つことを望むアカウントタイプを選択します。

This table lists the roles you can assign to a member of a team:

この表は、チームメンバーに割り当てることができる役割を示しています：

| Role   |   Definition   |
|-----------|---------------------------|
| admin    | A user who can add and remove other users in the team, change user roles, and configure team settings.   |
| Member    | A regular user of a team, invited by email or their organization-level username by the team admin. A member user cannot invite other users to the team.  |
| View-Only (Enterprise-only feature) | A view-only user of a team, invited by email or their organization-level username by the team admin. A view-only user only has read access to the team and its contents.  |
| Service (Enterprise-only feature)   | A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure to set the environment variable `WANDB_USERNAME`  to correctly attribute runs to the appropriate user. |
| Custom Roles (Enterprise-only feature)   | Custom roles allow organization admins to compose new roles by inheriting from the preceding View-Only or Member roles, and adding additional permissions to achieve fine-grained access control. Team admins can then assign any of those custom roles to users in their respective teams. Refer to [this article](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) for details. |

| 役割   |   定義   |
|-----------|---------------------------|
| 管理者    | チーム内で他のユーザーを追加し削除したり、ユーザー役割を変更したり、チームの設定を構成できるユーザー。   |
| メンバー  | チーム管理者によってメールまたは組織レベルのユーザー名で招待された、チームの通常のユーザー。メンバー ユーザーは、他のユーザーをチームに招待できません。  |
| ビュー専用 (エンタープライズのみの機能) | チーム管理者によってメールまたは組織レベルのユーザー名で招待された、チームのビュー専用ユーザー。ビュー専用ユーザーは、チームとそのコンテンツに対して読み取り専用アクセスしか持たない。  |
| サービス (エンタープライズのみの機能)   | サービスワーカーまたはサービスアカウントは、W&Bをあなたのrunオートメーションツールで利用するために役立つAPIキーです。チームのためにサービスアカウントからAPIキーを使用する場合は、環境変数 `WANDB_USERNAME` を設定して正しいユーザーにrunを紐付けることを確認してください。 |
| カスタムロール (エンタープライズのみの機能)   | カスタム役割は、組織管理者が前述の View-Only または Member 役割を継承し、追加の許可を追加して微細かいアクセス制御を達成するために作成することができます。チーム管理者は、その役割をそれぞれのチーム内のユーザーに割り当てることができます。詳細については、[この記事](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) を参照してください。 |

{{% alert %}}
Only enterprise licenses on Dedicated cloud or Self-managed deployment can assign custom roles to members in a team.
{{% /alert %}}

{{% alert %}}
専用クラウドまたは自己管理デプロイメントのエンタープライズライセンスのみが、チームメンバーにカスタム役割を割り当てることができます。
{{% /alert %}}

### Remove users from a team

### チームからユーザーを削除する

Remove a user from a team using the team's dashboard. W&B preserves runs created in a team even if the member who created the runs is no longer on that team.

チームのダッシュボードを使用してユーザーをチームから削除します。メンバーが作成したrunは、そのメンバーがそのチームにいなくなった場合でもW&Bに保存されます。

1. Navigate to `https://wandb.ai/<team-name>`.
2. Select **Team settings** in the left navigation bar.
3. Select the **Users** tab.
4. Hover your mouse next to the name of the user you want to delete. Select the ellipses or three dots icon (**...**) when it appears.
5. From the dropdown, select **Remove user**.

1. `https://wandb.ai/<team-name>` に移動します。
2. 左側のナビゲーションバーで **チーム設定** を選択します。
3. **ユーザー** タブを選択します。
4. 削除したいユーザーの名前の横にマウスをホバーします。表示されたら、三点リーダーまたは3つの点のアイコン（**...**）を選択します。
5. ドロップダウンから **ユーザーを削除** を選択します。