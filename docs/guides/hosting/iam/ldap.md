---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# LDAP を使用した SSO

W&B Server LDAP サーバーで資格情報を認証します。以下のガイドでは、W&B Server の設定方法を説明します。必須およびオプションの設定、システム設定 UI からの LDAP 接続の設定手順について説明しています。また、アドレス、ベース識別名、属性など、LDAP 設定のさまざまな入力についての情報も提供します。これらの属性は W&B アプリ UI または環境変数を使用して指定できます。匿名バインドを設定するか、管理者 DN と パスワード でバインドすることができます。

:::tip
LDAP 認証を有効にして設定できるのは W&B 管理者 ロールのみです。
:::

## LDAP 接続の設定

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'Environment variables', value: 'env'},
  ]}>
  <TabItem value="app">

1. W&B アプリに移動します。
2. 右上のプロフィールアイコンを選択します。ドロップダウンから **System Settings** を選択します。
3. **Configure LDAP Client** を切り替えます。
4. フォームに詳細を追加します。各入力項目の詳細については **Configuring Parameters** セクションを参照してください。
5. **Update Settings** をクリックして設定をテストします。これにより、W&B サーバーとのテスト クライアント/接続が確立されます。
6. 接続が確認された場合は、 **Enable LDAP Authentication** を切り替え、 **Update Settings** ボタンを選択します。

  </TabItem>
  <TabItem value="env">

以下の環境変数で LDAP 接続を設定します:

| 環境変数                    | 必須      | 例                                |
| ------------------------- | -------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`      | はい       | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`      | はい       | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`      | いいえ     | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`      | いいえ     |                                 |
| `LOCAL_LDAP_ATTRIBUTES`   | はい       | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`   | いいえ     |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | いいえ  |                                 |
| `LOCAL_LDAP_LOGIN`        | いいえ     |                                 |

各環境変数の定義については [Configuration parameters](#configuration-parameters) セクションを参照してください。明確にするために、環境変数のプレフィックス `LOCAL_LDAP` は定義名から省略されました。

  </TabItem>
</Tabs>

## 設定パラメータ

以下の表に、必須およびオプションの LDAP 設定を一覧し、説明します。

| 環境変数                  | 定義                                                                                                                                                    | 必須      |
| --------------------     | ----------------------------------------------------------------------------------------------------------------------------------------------------  | -------- |
| `ADDRESS`                | W&B ServerをホストするVPC内のLDAPサーバーのアドレスです。                                                                                              | はい       |
| `BASE_DN`                | このディレクトリー内のクエリを実行するために検索が開始されるルートパス。                                                                                | はい       |
| `BIND_DN`                | LDAPサーバーに登録された管理者ユーザーのパス。このパスは、LDAPサーバーが認証されていないバインドをサポートしていない場合に必要です。指定された場合、W&B サーバーはこのユーザーとしてLDAPサーバーに接続します。それ以外の場合、W&Bサーバーは匿名バインディングを使用して接続します。 | いいえ     |
| `BIND_PW`                | バインディングを認証するために使用される管理者ユーザーのパスワード。空白のままにすると、W&Bサーバーは匿名バインディングを使用して接続します。                                                 | いいえ     |
| `ATTRIBUTES`             | 電子メールとグループID属性名をカンマ区切りの文字列値として提供します。                                                                                 | はい       |
| `TLS_ENABLE`             | TLS を有効にします。                                                                                                                                  | いいえ     |
| `GROUP_ALLOW_LIST`       | グループの許可リスト。                                                                                                                                | いいえ     |
| `LOGIN`                  | W&B ServerにLDAPを使用して認証するように指示します。`True` または `False` のいずれかに設定します。LDAP設定をテストするためにオプションでこれをfalseに設定します。LDAP認証を開始するにはこれをtrueに設定します。                        | いいえ     |