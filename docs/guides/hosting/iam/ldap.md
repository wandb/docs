---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LDAPを使用したSSO

W&B ServerのLDAPサーバーで資格情報を認証します。次のガイドでは、W&B Serverの設定方法を説明します。このガイドには、必須およびオプションの設定、システム設定UIからのLDAP接続の設定方法が含まれています。また、アドレス、ベース識別名、属性などのLDAP設定のさまざまな入力に関する情報も提供します。これらの属性をW&B AppのUIから指定することも、環境変数を使用して指定することもできます。匿名バインド、または管理者DNおよびパスワードを使用するバインドのどちらかを設定できます。

:::tip
LDAP認証を有効化および設定できるのはW&B Adminロールのみです。
:::

## LDAP接続の設定

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'Environment variables', value: 'env'},
    
  ]}>
  <TabItem value="app">

1. W&B Appに移動します。
2. 右上のプロフィールアイコンを選択します。ドロップダウンから**System Settings**を選択します。
3. **Configure LDAP Client**をトグルします。
4. フォームに詳細を入力します。各入力の詳細については**Configuring Parameters**セクションを参照してください。
5. **Update Settings**をクリックして設定をテストします。これにより、W&Bサーバーとのテストクライアント/接続が確立されます。
6. 接続が確認されたら、**Enable LDAP Authentication**をトグルし、**Update Settings**ボタンを選択します。

  </TabItem>
  <TabItem value="env">

次の環境変数を使用してLDAP接続を設定します：

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

各環境変数の定義については[Configuration parameters](#configuration-parameters)セクションを参照してください。明確にするために、環境変数プレフィックス`LOCAL_LDAP`は定義名から省略されています。

  </TabItem>
</Tabs>

## 設定パラメーター

以下の表は、必要およびオプションのLDAP設定を一覧し、説明しています。

| Environment variable | 定義                                                                                                                                                                                                                                                                     | Required |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | --- |
| `ADDRESS`            | これは、W&B ServerをホストするVPC内のLDAPサーバーのアドレスです。                                                                                                                                                                                                      | Yes      |
| `BASE_DN`            | このディレクトリ内でクエリを実行するための検索の開始パスです。                                                                                                                                                                                                          | Yes      |
| `BIND_DN`            | LDAPサーバーに登録されている管理ユーザーのパス。LDAPサーバーが認証されていないバインディングをサポートしていない場合に必要です。指定された場合、W&B ServerはこのユーザーとしてLDAPサーバーに接続します。指定されていない場合、W&B Serverは匿名バインディングを使用します。 | No       |
| `BIND_PW`            | バインディングを認証するために使用される管理ユーザーのパスワード。空白のままにすると、W&B Serverは匿名バインディングを使用します。                                                                                                                                        | No       |     |
| `ATTRIBUTES`         | メールとグループIDの属性名をカンマ区切りの文字列値として指定します。                                                                                                                                                                                                  | Yes      |
| `TLS_ENABLE`         | TLSを有効にします。                                                                                                                                                                                                                                                     | No       |
| `GROUP_ALLOW_LIST`   | グループの許可リスト。                                                                                                                                                                                                                                                  | No       |
| `LOGIN`              | W&B ServerにLDAPを使って認証するよう指示します。`True`または`False`のいずれかを設定します。オプションで、LDAP設定をテストするためにfalseに設定できます。LDAP認証を開始するにはtrueに設定します。                                                                            | No       |