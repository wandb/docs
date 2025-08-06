---
title: LDAP で SSO を設定
menu:
  default:
    identifier: ldap
    parent: authentication
---

W&B Server の LDAP サーバーで認証情報を認証します。以下のガイドでは、W&B Server の設定方法について説明します。必須およびオプションの設定、そしてシステム設定 UI からの LDAP 接続の設定方法が含まれます。また、アドレス、ベース識別名、属性など LDAP 設定のさまざまな入力項目についての情報も提供します。これらの属性は W&B App の UI または環境変数で指定できます。匿名バインド、または管理者 DN とパスワードでのバインドのいずれかを設定できます。

{{% alert %}}
LDAP認証の有効化・設定は W&B Admin ロールのみ可能です。
{{% /alert %}}

## LDAP 接続の設定

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App にアクセスします。 
2. 右上のプロフィールアイコンを選択し、ドロップダウンから **System Settings** を選びます。
3. **Configure LDAP Client** をトグルで有効にします。
4. フォームに詳細情報を入力します。各項目の詳細は**パラメータの設定**セクションを参照してください。
5. **Update Settings** をクリックして設定をテストします。これで W&B Server とのテストクライアント／接続が確立されます。
6. 接続が検証されたら、**Enable LDAP Authentication** をトグルで有効にし、**Update Settings** ボタンを選択します。
{{% /tab %}}

{{% tab header="環境変数" value="env"%}}
以下の環境変数を使って、LDAP 接続を設定します:

| 環境変数                   | 必須     | 例                                  |
| -------------------------- | -------- | ----------------------------------- |
| `LOCAL_LDAP_ADDRESS`       | はい     | `ldaps://ldap.example.com:636`      |
| `LOCAL_LDAP_BASE_DN`       | はい     | `email=mail,group=gidNumber`        |
| `LOCAL_LDAP_BIND_DN`       | いいえ   | `cn=admin`, `dc=example,dc=org`     |
| `LOCAL_LDAP_BIND_PW`       | いいえ   |                                     |
| `LOCAL_LDAP_ATTRIBUTES`    | はい     | `email=mail`, `group=gidNumber`     |
| `LOCAL_LDAP_TLS_ENABLE`    | いいえ   |                                     |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | いいえ |                                     |
| `LOCAL_LDAP_LOGIN`         | いいえ   |                                     |

各環境変数の定義については[設定パラメータ]({{< relref "#configuration-parameters" >}}) セクションを参照してください。分かりやすくするため、環境変数名の先頭接頭辞 `LOCAL_LDAP` は定義名から省略しています。
{{% /tab %}}
{{< /tabpane >}}


## 設定パラメータ

以下の表は、LDAP 設定項目の必須・任意と説明です。

| 環境変数        | 定義                                               | 必須     |
| --------------- | -------------------------------------------------- | -------- |
| `ADDRESS`       | W&B Server がある VPC 内の LDAP サーバーのアドレス | はい     |
| `BASE_DN`       | クエリの起点となるディレクトリのルートパス          | はい     |
| `BIND_DN`       | LDAP サーバーに登録された管理者ユーザーのパス。この項目は LDAP サーバーが未認証のバインドをサポートしない場合に必須です。指定した場合、W&B Server はこのユーザーとして LDAP サーバーに接続します。未指定の場合は匿名バインドを利用します。 | いいえ   |
| `BIND_PW`       | 管理者ユーザーのパスワードで、バインドの認証に使用します。空欄の場合、W&B Server は匿名バインドを利用します。 | いいえ   |
| `ATTRIBUTES`    | メールとグループID の属性名をカンマ区切りで指定します。     | はい     |
| `TLS_ENABLE`    | TLS を有効にします。                                 | いいえ   |
| `GROUP_ALLOW_LIST` | 許可するグループのリスト。                      | いいえ   |
| `LOGIN`         | LDAP を使用して W&B Server の認証を行うかどうか指定します。`True` または `False` を設定します。LDAP 設定のテスト用に任意で False にできます。LDAP 認証を開始する場合は True にしてください。 | いいえ   |