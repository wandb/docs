---
displayed_sidebar: default
---
# ユーザー管理

Weights & Biasesは、シングルサインオン（SSO）を介したユーザー管理を強く推奨しています。W&BサーバーでSSOを設定する方法については、[SSO設定ドキュメント](./sso.md)を参照してください。

:::tip
W&Bを使用する際、ユーザーは_admin_または_member_のいずれかに分類されます。管理者は他の管理者やチームメンバーを追加・削除することができます。チームメンバーは、チーム管理者からメールで招待されます。チームメンバーは、他のメンバーを招待することはできません。

詳細については、[チームの役割と権限を確認](../app/features/teams#team-roles-and-permissions)してください。
:::

## インスタンス管理者

W&Bサーバーをデプロイした後に最初にサインアップするユーザーは、自動的に管理者権限が割り当てられます。管理者は、インスタンスに追加のユーザーを追加し、チームを作成することができます。

## ユーザーを招待する

`https://<YOUR-WANDB-URL>/admin/users`ページから、他の管理者やメンバーを招待してください。

1. `https://<YOUR-WANDB-URL>/admin/users`にアクセスします。

![](/images/hosting/invite_users.png)

2. **Add User**をクリックします。

![](/images/hosting/add_user_empty_field.png)

3. ユーザーのメールアドレスを入力します。デフォルトでは、すべてのユーザーはMembersとして招待されます。インスタンスAdminとして誰かを招待する必要がある場合は、**Admin option**を切り替えて**Submit**をクリックします。
![](/images/hosting/add_user_field_filled.png)

<!-- ![Screen Shot 2023-01-09 at 10.16.04 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1428275-5ae0-4a36-8c1b-99248d7a7584/Screen_Shot_2023-01-09_at_10.16.04_PM.png) -->

招待リンクは、ユーザーにメールで送信されます。新しい管理者またはメンバーは、W＆Bインスタンスにアクセスできるようになります。

W&Bは、これらの招待メールを送信するために、サードパーティのメールサーバーを使用しています。組織のファイアウォールルールが企業ネットワーク外へのトラフィックの送信を禁止している場合、W&Bは内部のSMTPサーバーを設定するオプションを提供します。SMTPサーバーの設定方法については、[これらの手順](./smtp.md) を参照してください。

<!-- To do: Add this doc -->
<!-- Refer to SMTP configuration documentation for instructions on how to do this. -->

## チームを作成する

`https://<YOUR-WANDB-URL>/admin/users` に移動して、新しいW＆Bチームを作成します。

1. `https://<YOUR-WANDB-URL>/admin/users` ページに移動し、**チーム** をクリックします。

![](/images/hosting/manage_users_teams.png)

<!-- ![Screen Shot 2023-01-09 at 10.22.50 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7d59520c-4a00-4596-9e2e-428b1b53c589/Screen_Shot_2023-01-09_at_10.22.50_PM.png) -->

2. **新しいチーム** をクリックし、**チーム名** フィールドにチーム名を入力します。

![](/images/hosting/manage_users_teams_filled.png)

<!-- ![Screen Shot 2023-01-09 at 10.25.10 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/180f26ae-fa96-4dc4-b421-f9676ff73477/Screen_Shot_2023-01-09_at_10.25.10_PM.png) -->

各チームには、独自のプロフィールページがあります。`https://<YOUR-WANDB-URL>/<team-name>` に移動して、チームのプロフィールページを表示します。

![](/images/hosting/add_teams_server.png)
<!-- ![Screen Shot 2023-01-09 at 10.29.14 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7dbd7cac-9300-4a48-a67c-a696548b0153/Screen_Shot_2023-01-09_at_10.29.14_PM.png) -->

## チーム設定の管理

チームホームページにはチーム設定のオプションが含まれており、メンバーを管理したり、チームのアバターを設定したり、プライバシー設定を調整したり、アラートを設定したり、使用状況をトラッキングしたり、その他の操作を行うことができます。詳細については、[チーム設定](../app/settings-page/team-settings.md)ページをご覧ください。

## チームにメンバーを招待する

:::info
メンバーは、チームに招待される前にインスタンスの一部でなければなりません。インスタンスにユーザーを招待する方法については、[ユーザーの招待](#invite-users)セクションを参照してください。
:::

チームにユーザーを招待する際に、以下の役割のいずれかを割り当てることができます。

| 役割    | 定義                                                                                                                                                                                                                                                                                       |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 管理者   | 他の管理者やチームのメンバーを追加・削除できるチームメンバー。                                                                                                                                                                                                                       |
| メンバー  | チーム管理者がメールで招待したチームの通常メンバー。チームメンバーは他のメンバーをチームに招待することはできません。                                                                                                                                                                        |
| サービス | サービスワーカーやサービスアカウントは、W&Bとランの自動化ツールを連携させる際に役立つAPIキーです。チーム用のサービスアカウントからAPIキーを使用する場合は、環境変数`WANDB_USERNAME`を設定して、runsを適切なユーザーに正しく割り当てるようにしてください。 |

![](/images/hosting/team_settings_wand_server_example.png)

<!-- **Admin**: A team member who can add and remove other admins and members of the team.

**Member**: A regular member of your team, invited by email by the team admin. A team member cannot invite other members to the team.

**Service**: A service worker or service account is an API key that is useful for utilizing W&B with your run automation tools. If you use an API key from a service account for your team, ensure that the environment variable `WANDB_USERNAME` is set to correctly attribute runs to the appropriate user. -->

<!-- ![Screen Shot 2023-01-09 at 10.48.49 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2eb67576-e0c5-4951-95ba-7a6fa49a8d68/Screen_Shot_2023-01-09_at_10.48.49_PM.png) -->
## チームからメンバーを削除する



チームの設定ページを使用して、メンバーを削除してください。



1. チーム設定ページに移動します。

2. メンバーの名前の隣にある削除ボタンを選択します。



:::info

チームメンバーが削除された後も、W&B runsはログに残ります。

:::