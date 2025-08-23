---
title: LDAP で SSO を設定
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-ldap
    parent: authentication
---

W&B サーバーの LDAP サーバーで認証情報を認証します。以下のガイドでは、W&B サーバーの設定方法について説明します。必須およびオプションの設定内容、システム設定 UI からの LDAP 接続設定方法、アドレスやベースディスティングイッシュドネーム、属性など LDAP 設定の各入力項目についての情報も提供しています。これらの属性は、W&B App UI から、または環境変数を使って指定できます。匿名バインド、または管理者 DN とパスワードによるバインドのいずれかを設定できます。

{{% alert %}}
W&B Admin ロールのみが LDAP 認証の有効化と設定を行うことができます。
{{% /alert %}}

## LDAP 接続の設定

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App に移動します。 
2. 画面右上のプロフィールアイコンをクリックし、ドロップダウンから **System Settings** を選択します。 
3. **Configure LDAP Client** を有効にします。
4. フォームに詳細情報を入力します。各入力項目の詳細は **Configuring Parameters** セクションを参照してください。
5. **Update Settings** をクリックして設定をテストします。これにより、W&B サーバーとのテストクライアント/接続が確立されます。
6. 接続が検証されたら、**Enable LDAP Authentication** を切り替えて **Update Settings** ボタンを選択します。
{{% /tab %}}

{{% tab header="Environment variable" value="env"%}}
以下の環境変数を使用して LDAP 接続を設定します:

| 環境変数          | 必須 | 例                         |
| ----------------------------- | -------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`          | はい      | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`          | はい      | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`          | いいえ   | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`          | いいえ   |                                 |
| `LOCAL_LDAP_ATTRIBUTES`       | はい      | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`       | いいえ   |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | いいえ   |                                 |
| `LOCAL_LDAP_LOGIN`            | いいえ   |                                 |

各環境変数の定義については、[Configuration parameters]({{< relref path="#configuration-parameters" lang="ja" >}}) セクションを参照してください。なお、定義名の説明をわかりやすくするため、環境変数のプレフィックス `LOCAL_LDAP` は省略しています。
{{% /tab %}}
{{< /tabpane >}}


## Configuration parameters

以下の表は、必須およびオプションの LDAP 設定内容の一覧と説明です。

| 環境変数 | 定義              | 必須 |
| -------------------- | ----------------------- | -------- |
| `ADDRESS`            | W&B Server をホストしている VPC 内の LDAP サーバーのアドレスです。      | はい      |
| `BASE_DN`            | 検索が開始されるルートパスです。このディレクトリー内でのクエリ実行に必須です。             | はい      |
| `BIND_DN`            | LDAP サーバーに登録された管理ユーザーのパスです。LDAP サーバーが未認証バインドをサポートしない場合に必要です。指定した場合は、このユーザーとして W&B Server が LDAP サーバーに接続します。指定しない場合は、匿名バインドで接続します。 | いいえ       |
| `BIND_PW`            | 管理ユーザーのパスワードです。バインドの認証に使用されます。空欄の場合は匿名バインドで接続します。   | いいえ       |
| `ATTRIBUTES`         | メールアドレスおよびグループ ID の属性名をカンマ区切り文字列で指定します。   | はい      |
| `TLS_ENABLE`         | TLS を有効化します。                | いいえ       |
| `GROUP_ALLOW_LIST`   | グループの許可リストです。           | いいえ       |
| `LOGIN`              | W&B Server が LDAP で認証を行うかどうかを指定します。`True` または `False` のいずれかをセット可能です。設定のテスト用に false とすることもできます。LDAP 認証を開始する場合は true にします。 | いいえ       |