---
title: Configure SSO with LDAP
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-ldap
    parent: authentication
---

W&B Server LDAP サーバー で認証情報 を認証します。次のガイドでは、W&B Server の 設定 を構成する方法について説明します。必須およびオプションの 設定 、システム 設定 UI から LDAP 接続 を構成する手順について説明します。また、 アドレス 、ベース識別名、 属性 など、LDAP 構成 のさまざまな入力に関する情報も提供します。これらの 属性 は、W&B App UI から、または 環境変数 を使用して指定できます。匿名バインド、または管理者 DN とパスワードを使用してバインドを設定できます。

{{% alert %}}
W&B 管理者 ロール のみが LDAP 認証を有効化および構成できます。
{{% /alert %}}

## LDAP 接続 の構成

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App に移動します。
2. 右上からプロファイル アイコンを選択します。ドロップダウンから、**システム 設定** を選択します。
3. **LDAPクライアント を構成** を切り替えます。
4. フォームに詳細を追加します。各入力の詳細については、**パラメータ の構成** セクションを参照してください。
5. **設定 の更新** をクリックして、 設定 をテストします。これにより、W&B サーバー とのテストクライアント/ 接続 が確立されます。
6. 接続 が検証されたら、**LDAP 認証 を有効にする** を切り替えて、**設定 の更新** ボタンを選択します。
{{% /tab %}}

{{% tab header="Environment variable" value="env"%}}
次の 環境変数 を使用して LDAP 接続 を設定します。

| 環境変数                      | 必須 | 例                               |
| ----------------------------- | -------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`          | はい      | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`          | はい      | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`          | いいえ       | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`          | いいえ       |                                 |
| `LOCAL_LDAP_ATTRIBUTES`       | はい      | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`       | いいえ       |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | いいえ       |                                 |
| `LOCAL_LDAP_LOGIN`            | いいえ       |                                 |

各 環境変数 の定義については、[設定 パラメータ]({{< relref path="#configuration-parameters" lang="ja" >}}) セクションを参照してください。わかりやすくするために、 環境変数 のプレフィックス `LOCAL_LDAP` は定義名から省略されていることに注意してください。
{{% /tab %}}
{{< /tabpane >}}

## 設定 パラメータ

次の表に、必須およびオプションの LDAP 構成 を示します。

| 環境変数   | 定義                                                              | 必須 |
| -------------------- | ---------------------------------------------------------------- | -------- |
| `ADDRESS`            | これは、W&B Server をホストする VPC 内の LDAP サーバー の アドレス です。                                   | はい      |
| `BASE_DN`            | ルートパス検索の開始元であり、この ディレクトリー へのクエリを実行するために必要です。                               | はい      |
| `BIND_DN`            | LDAP サーバー に登録されている管理 ユーザー のパス。LDAP サーバー が認証されていないバインディングをサポートしていない場合に必要です。指定した場合、W&B Server はこの ユーザー として LDAP サーバー に 接続 します。それ以外の場合、W&B Server は匿名バインディングを使用して 接続 します。 | いいえ       |
| `BIND_PW`            | 管理 ユーザー のパスワード。これはバインディングの認証に使用されます。空白のままにすると、W&B Server は匿名バインディングを使用して 接続 します。                         | いいえ       |
| `ATTRIBUTES`         | カンマ区切りの文字列 値 として、メール アドレス とグループ ID 属性 名を指定します。                                 | はい      |
| `TLS_ENABLE`         | TLS を有効にします。                                                            | いいえ       |
| `GROUP_ALLOW_LIST`   | グループ 許可リスト。                                                           | いいえ       |
| `LOGIN`              | これは、W&B Server に LDAP を使用して認証するように指示します。`True` または `False` に設定します。オプションで、LDAP 構成 をテストするためにこれを false に設定します。LDAP 認証 を開始するには、これを true に設定します。             | いいえ       |
