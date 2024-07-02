---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSO using LDAP

W&B Server の LDAP サーバーで認証情報を認証します。以下のガイドでは、W&B Server の設定方法を説明します。必須およびオプションの設定、システム設定 UI からの LDAP 接続の設定方法について説明します。また、アドレス、base distinguished name、および属性など、LDAP 設定のさまざまな入力についての情報も提供します。これらの属性は、W&B App UI からまたは環境変数を使用して指定できます。匿名バインド、または管理者 DN とパスワードでバインドを設定できます。

:::tip
LDAP 認証を有効にして設定できるのは、W&B 管理者ロールのみです。
:::

## LDAP 接続の設定

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'Environment variables', value: 'env'},
  ]}>
  <TabItem value="app">

1. W&B App に移動します。
2. 右上のプロフィールアイコンを選択します。ドロップダウンから **System Settings** を選択します。
3. **Configure LDAP Client** をトグルします。
4. フォームに詳細を入力します。各入力の詳細については **Configuring Parameters** セクションを参照してください。
5. **Update Settings** をクリックして設定をテストします。これにより、W&B サーバーとのテストクライアント/接続が確立されます。
6. 接続が確認されたら、**Enable LDAP Authentication** をトグルし、**Update Settings** ボタンを選択します。


  </TabItem>
  <TabItem value="env">

以下の環境変数を使用して LDAP 接続を設定します：

| Environment variable          | Required | Example                         |
| ----------------------------- | -------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`          | Yes      | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`          | Yes      | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`          | No       | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`          | No       |                                 |
| `LOCAL_LDAP_ATTRIBUTES`       | Yes      | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`       | No       |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | No       |                                 |
| `LOCAL_LDAP_LOGIN`            | No       |                                 |

各環境変数の定義については [Configuration parameters](#configuration-parameters) セクションを参照してください。環境変数のプレフィックス `LOCAL_LDAP` は明確にするために省略されている点に注意してください。

  </TabItem>
</Tabs>

## 設定パラメータ

以下の表は、必須およびオプションの LDAP 設定を一覧にして説明しています。

| Environment variable | Definition                                                                                                                                                                                                                                                              | Required |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `ADDRESS`            | これは、W&B Server がホストされている VPC 内の LDAP サーバーのアドレスです。                                                                                                                                                                                           | Yes      |
| `BASE_DN`            | 検索が開始するルートパスで、このディレクトリーへのクエリを行うために必要です。                                                                                                                                                                               | Yes      |
| `BIND_DN`            | LDAP サーバーに登録されている管理者ユーザーのパスです。LDAP サーバーが未認証のバインドをサポートしていない場合に必要です。指定されている場合、W&B Server はこのユーザーとして LDAP サーバーに接続します。それ以外の場合、W&B Server は匿名バインドを使用して接続します。 | No       |
| `BIND_PW`            | 管理者ユーザーのパスワードで、バインドを認証するために使用されます。空白のままにすると、W&B Server は匿名バインドを使用して接続します。                                                                                                                             | No       |
| `ATTRIBUTES`         | email およびグループ ID 属性名をカンマ区切りの文字列値として提供します。                                                                                                                                                                                         | Yes      |
| `TLS_ENABLE`         | TLS を有効にします。                                                                                                                                                                                                                                                             | No       |
| `GROUP_ALLOW_LIST`   | グループの許可リストです。                                                                                                                                                                                                                                                        | No       |
| `LOGIN`              | これは、W&B Server が認証に LDAP を使用することを指示します。`True` か `False` に設定します。オプションで LDAP 設定をテストするために false に設定することもできます。LDAP 認証を開始するには true に設定します。                                                                         | No       |