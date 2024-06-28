---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# LDAP を使用した SSO

W&B Server の LDAP サーバーで資格情報を認証します。以下のガイドでは、W&B Server の設定方法について説明します。必須およびオプションの設定、システム設定 UI からの LDAP 接続の設定方法について詳しく説明します。また、アドレス、基本識別名、属性など、LDAP 設定のさまざまな入力についての情報も提供します。これらの属性は W&B App UI から、または環境変数を使用して指定できます。匿名バインド、または管理者 DN とパスワードを使用したバインドのいずれかを設定できます。

:::tip
LDAP 認証を有効にして設定できるのは W&B 管理者ロールのみです。
:::

## LDAP 接続の設定

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: '環境変数', value: 'env'},
    
  ]}>
  <TabItem value="app">

1. W&B App に移動します。 
2. 右上のプロフィールアイコンを選択します。ドロップダウンから **System Settings** を選択します。
3. **Configure LDAP Client** をトグルします。
4. フォームに詳細を入力します。各入力の詳細については **Configuring Parameters** セクションを参照してください。
5. **Update Settings** をクリックして設定をテストします。これにより、W&B サーバーとのテストクライアント/接続が確立されます。
6. 接続が確認された場合は、**Enable LDAP Authentication** をトグルして **Update Settings** ボタンを選択します。

  </TabItem>
  <TabItem value="env">

次の環境変数を使用して LDAP 接続を設定します：

| 環境変数                   | 必須    | 例                             |
| --------------------------- | ------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`        | はい    | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`        | はい    | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`        | いいえ  | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`        | いいえ  |                                 |
| `LOCAL_LDAP_ATTRIBUTES`     | はい    | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`     | いいえ  |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST`| いいえ |                                 |
| `LOCAL_LDAP_LOGIN`          | いいえ  |                                 |

各環境変数の定義については [Configuration parameters](#configuration-parameters) セクションを参照してください。明確にするために、定義名から環境変数接頭辞 `LOCAL_LDAP` を省略しました。

  </TabItem>
</Tabs>

## 設定パラメータ

以下の表は必須およびオプションの LDAP 設定について説明しています。

| 環境変数       | 定義                                                                                                                                                                                                                                                                           | 必須    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | 
| `ADDRESS`      | これは VPC 内の W&B Server をホストする LDAP サーバーのアドレスです。                                                                                                                                                                                                         | はい    |
| `BASE_DN`      | このディレクトリーに対して任意のクエリーを実行するために必要なルートパスです。                                                                                                                                                                                                   | はい    |
| `BIND_DN`      | LDAP サーバーに登録されている管理ユーザーのパスです。LDAP サーバーが認証なしバインディングをサポートしない場合に必要です。指定されている場合、W&B Server はこのユーザーとして LDAP サーバーに接続します。そうでない場合、W&B Server は匿名バインディングを使用して接続します。                | いいえ  |
| `BIND_PW`      | 管理ユーザーのパスワードで、バインディングの認証に使用されます。空白の場合、W&B Server は匿名バインディングを使用して接続します。                                                                                                                                                 | いいえ  |
| `ATTRIBUTES`   | カンマ区切りの文字列値としてメールアドレスおよびグループ ID 属性名を提供します。                                                                                                                                                                                               | はい    |
| `TLS_ENABLE`   | TLS を有効にします。                                                                                                                                                                                                                                                             | いいえ  |
| `GROUP_ALLOW_LIST` | グループ許可リスト。                                                                                                                                                                                                                                                            | いいえ  |
| `LOGIN`        | W&B Server に LDAP を使用して認証するよう指示します。`True` または `False` に設定します。オプションとしてこの値を `False` に設定して LDAP 設定をテストできます。`True` に設定して LDAP 認証を開始します。                                                                        | いいえ  |