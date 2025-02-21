---
title: Configure SSO with LDAP
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-ldap
    parent: authentication
---

あなたの資格情報を W&B サーバーの LDAP サーバーで認証します。以下のガイドは、W&B サーバーの設定を行う方法を説明します。必須およびオプションの設定、システム設定 UI からの LDAP 接続の設定方法に関する指示を網羅しています。また、アドレス、ベース識別名、および属性など、LDAP 設定のさまざまな入力方法に関する情報も提供しています。これらの属性は W&B アプリ UI から指定するか、環境変数を使用して指定することができます。匿名バインド、または管理者 DN とパスワードを用いたバインドを設定できます。

{{% alert %}}
LDAP 認証を有効にし、設定できるのは W&B 管理者ロールのみです。
{{% /alert %}}

## LDAP 接続の設定

{{< tabpane text=true >}}
{{% tab header="W&B アプリ" value="app" %}}
1. W&B アプリに移動します。
2. 右上のプロフィールアイコンを選択します。ドロップダウンから **System Settings** を選択します。
3. **Configure LDAP Client** を切り替えます。
4. フォームに詳細を入力します。各入力に関する詳細については、**Configuring Parameters** セクションを参照してください。
5. **Update Settings** をクリックして設定をテストします。これにより、W&B サーバーとのテストクライアント/接続が確立されます。
6. 接続が検証済みである場合、**Enable LDAP Authentication** を切り替え、**Update Settings** ボタンを選択します。
{{% /tab %}}

{{% tab header="環境変数" value="env"%}}
次の環境変数を使用して LDAP 接続を設定します。

| 環境変数                        | 必須   | 例                                |
| ----------------------------- | ------ | -------------------------------- |
| `LOCAL_LDAP_ADDRESS`          | Yes    | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`          | Yes    | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`          | No     | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`          | No     |                                 |
| `LOCAL_LDAP_ATTRIBUTES`       | Yes    | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`       | No     |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | No     |                                 |
| `LOCAL_LDAP_LOGIN`            | No     |                                 |

各環境変数の定義については [Configuration parameters]({{< relref path="#configuration-parameters" lang="ja" >}}) セクションを参照してください。わかりやすくするために、環境変数の定義名から `LOCAL_LDAP` プレフィックスを省略しています。
{{% /tab %}}
{{< /tabpane >}}

## 設定パラメータ

以下のテーブルでは、必須およびオプションの LDAP 設定を一覧化して説明します。

| 環境変数       | 定義                        | 必須   |
| -------------- | --------------------------- | ------ |
| `ADDRESS`      | W&B サーバーをホストする VPC 内の LDAP サーバーのアドレスです。   | Yes    |
| `BASE_DN`      | ディレクトリー内のクエリを実行するために開始するルートパスです。  | Yes    |
| `BIND_DN`      | LDAP サーバーに登録されている管理ユーザーのパスです。LDAP サーバーが未認証のバインドをサポートしていない場合に必要です。指定されている場合、W&B サーバーはこのユーザーとして LDAP サーバーに接続します。そうでない場合、W&B サーバーは匿名バインドを使用して接続します。 | No     |
| `BIND_PW`      | 管理ユーザーのパスワードであり、バインドの認証に使用されます。空白の場合、W&B サーバーは匿名バインドを使用して接続します。 | No     |
| `ATTRIBUTES`   | メールとグループ ID の属性名をカンマ区切りの文字列として提供します。 | Yes    |
| `TLS_ENABLE`   | TLS を有効にします。          | No     |
| `GROUP_ALLOW_LIST` | グループ許可リスト。      | No     |
| `LOGIN`        | W&B サーバーに LDAP を使用して認証するよう指示します。`True` または `False` に設定します。オプションで、LDAP 設定をテストするために false に設定します。LDAP 認証を開始するには true に設定します。| No     |