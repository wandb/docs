---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSO using LDAP

W&B サーバー LDAP サーバーで資格情報を認証します。次のガイドでは、W&B Server の設定方法について説明します。必須およびオプションの設定に加えて、システム設定 UI からの LDAP 接続の設定方法についても説明します。また、アドレス、基本識別名、属性など、LDAP 設定のさまざまな入力に関する情報も提供します。これらの属性は W&B アプリ UI から指定するか、環境変数を使用して指定できます。匿名バインドか、管理者 DN とパスワードを使用したバインドのどちらかを設定できます。

:::tip
LDAP 認証を有効にして設定できるのは W&B 管理者ロールのみです。
:::

## LDAP 接続の設定

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B アプリ', value: 'app'},
    {label: '環境変数', value: 'env'},
    
  ]}>
  <TabItem value="app">

1. W&B アプリに移動します。
2. 右上にあるプロフィールアイコンを選択します。ドロップダウンから **System Settings** を選択します。
3. **Configure LDAP Client** を切り替えます。
4. フォームに詳細を追加します。各入力の詳細については **Configuring Parameters** セクションを参照してください。
5. **Update Settings** をクリックして設定をテストします。これにより、W&B サーバーとのテストクライアント/接続が確立されます。
6. 接続が確認されたら、**Enable LDAP Authentication** を切り替え、**Update Settings** ボタンを選択します。

  </TabItem>
  <TabItem value="env">

次の環境変数を使用して LDAP 接続を設定します:

| 環境変数                   | 必須       | 例                             |
| -------------------------- | ------- | ----------------------------- |
| `LOCAL_LDAP_ADDRESS`       | はい    | `ldaps://ldap.example.com:636` |
| `LOCAL_LDAP_BASE_DN`       | はい    | `email=mail,group=gidNumber`   |
| `LOCAL_LDAP_BIND_DN`       | いいえ  | `cn=admin`, `dc=example,dc=org`|
| `LOCAL_LDAP_BIND_PW`       | いいえ  |                                |
| `LOCAL_LDAP_ATTRIBUTES`    | はい    | `email=mail,group=gidNumber`   |
| `LOCAL_LDAP_TLS_ENABLE`    | いいえ  |                                |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | いいえ  |                                 |
| `LOCAL_LDAP_LOGIN`         | いいえ  |                                |

各環境変数の定義については、[Configuration parameters](#configuration-parameters) セクションを参照してください。明確にするために、環境変数のプレフィックス `LOCAL_LDAP` は定義名から省略されています。

  </TabItem>
</Tabs>

## 設定パラメータ

次の表に、必須およびオプションの LDAP 設定を一覧し、説明します。

