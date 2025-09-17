---
title: LDAP と SSO を設定する
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-ldap
    parent: authentication
---

W&B Server は LDAP サーバーで資格情報を認証します。以下のガイドでは、W&B Server の LDAP 設定方法を説明します。必須およびオプションの設定に加えて、System Settings UI からの LDAP 接続の設定手順を示します。また、アドレス、ベース DN (base distinguished name)、属性など、LDAP 設定で使用する各入力項目についても説明します。これらの属性は、W&B App の UI から、または環境変数で指定できます。匿名バインド (anonymous bind) または管理者 DN とパスワードでのバインドを設定できます。

{{% alert %}}
W&B の管理者ロールのみが LDAP 認証を有効化および設定できます。
{{% /alert %}}

## LDAP 接続の設定

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1.  W&B App に移動します。
2.  右上隅のプロフィールアイコンを選択し、ドロップダウンから「**System Settings**」を選択します。
3.  「**Configure LDAP Client**」を切り替えます。
4.  フォームに必要項目を入力します。各入力の詳細は「**構成パラメータ**」セクションを参照してください。
5.  「**Update Settings**」をクリックして設定をテストします。これにより、W&B Server とのテストクライアント/接続が確立されます。
6.  接続が検証されたら「**Enable LDAP Authentication**」を切り替え、「**Update Settings**」ボタンを選択します。
{{% /tab %}}

{{% tab header="環境変数" value="env"%}}
以下の環境変数を使用して LDAP 接続を設定します。

| 環境変数                      | 必須 | 例                              |
| :---------------------------- | :--- | :------------------------------ |
| `LOCAL_LDAP_ADDRESS`          | はい | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`          | はい | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`          | いいえ | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`          | いいえ |                                 |
| `LOCAL_LDAP_ATTRIBUTES`       | はい | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`       | いいえ |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | いいえ |                                 |
| `LOCAL_LDAP_LOGIN`            | いいえ |                                 |

各環境変数の定義については、「[構成パラメータ]({{< relref path="#configuration-parameters" lang="ja" >}})」セクションを参照してください。分かりやすくするため、定義名から環境変数プレフィックス `LOCAL_LDAP` は省略されています。
{{% /tab %}}
{{< /tabpane >}}

## 構成パラメータ

次の表に、必須およびオプションの LDAP 設定を示し、説明します。

| 環境変数             | 定義                                                                                                                                                                                                                                                                                            | 必須 |
| :------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--- |
| `ADDRESS`            | これは、W&B Server をホストしている VPC 内の LDAP サーバーの アドレス です。                                                                                                                                                                                                                     | はい |
| `BASE_DN`            | 検索の起点となるルートパスで、このディレクトリでクエリを実行するために必要です。                                                                                                                                                                                                                | はい |
| `BIND_DN`            | LDAP サーバーに登録されている管理者 ユーザー の DN です。LDAP サーバーが匿名バインドをサポートしない場合に必要です。指定されている場合、W&B Server はこの ユーザー として LDAP サーバーに接続し、指定がない場合は匿名バインドで接続します。                                                      | いいえ |
| `BIND_PW`            | 管理者 ユーザー のパスワードです。バインドの認証に使用されます。空白のままにした場合、W&B Server は匿名バインドを使用して接続します。                                                                                                                                                            | いいえ |
| `ATTRIBUTES`         | メール アドレスとグループ ID の属性名を、カンマ区切りの文字列の 値 として指定します。                                                                                                                                                                                                            | はい |
| `TLS_ENABLE`         | TLS を有効にします。                                                                                                                                                                                                                                                                              | いいえ |
| `GROUP_ALLOW_LIST`   | グループの許可リストです。                                                                                                                                                                                                                                                                        | いいえ |
| `LOGIN`              | W&B Server に LDAP 認証を使うかどうかを指示します。`True` または `False` に設定します。設定をテストする間は `False` に、LDAP 認証を開始するには `True` に設定します。                                                                                                                              | いいえ |