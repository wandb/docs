---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LDAP を使った SSO

W&B Server LDAP サーバーで資格情報を認証します。以下のガイドでは、W&B Server の設定方法を説明しています。必須設定とオプション設定、およびシステム設定 UI から LDAP 接続を設定する手順について説明します。また、アドレス、ベース識別名、および属性などの LDAP 設定のさまざまな入力項目に関する情報も提供します。これらの属性は、W&B アプリ UI または環境変数を使用して指定できます。匿名バインドを設定するか、管理者 DN とパスワードを使用してバインドすることができます。

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
2. 右上のプロフィールアイコンを選択します。ドロップダウンから**システム設定**を選択します。
3. **LDAP クライアントの設定**を切り替えます。
4. フォームに詳細を追加します。各入力項目の詳細については**設定パラメータ**セクションを参照してください。
5. **設定の更新**をクリックして設定をテストします。これにより、W&B サーバーとのテストクライアント/接続が確立されます。
6. 接続が確認されたら、**LDAP 認証を有効にする**を切り替え、**設定の更新**ボタンを選択します。

  </TabItem>
  <TabItem value="env">

以下の環境変数を使用して LDAP 接続を設定します:

| 環境変数                    | 必須  | 例                                  |
| --------------------------- | ---- | ---------------------------------- |
| `LOCAL_LDAP_ADDRESS`        | はい  | `ldaps://ldap.example.com:636`     |
| `LOCAL_LDAP_BASE_DN`        | はい  | `email=mail,group=gidNumber`        |
| `LOCAL_LDAP_BIND_DN`        | いいえ| `cn=admin`, `dc=example,dc=org`     |
| `LOCAL_LDAP_BIND_PW`        | いいえ|                                    |
| `LOCAL_LDAP_ATTRIBUTES`     | はい  | `email=mail`, `group=gidNumber`     |
| `LOCAL_LDAP_TLS_ENABLE`     | いいえ|                                    |
| `LOCAL_LDAP_GROUP_ALLOW_LIST`| いいえ|                                    |
| `LOCAL_LDAP_LOGIN`          | いいえ|                                    |

各環境変数の定義については、[設定パラメータ](#configuration-parameters) セクションを参照してください。明確にするために、環境変数のプレフィックス `LOCAL_LDAP` は定義名から省略されています。

  </TabItem>
</Tabs>

## 設定パラメータ

以下の表は、必須およびオプションの LDAP 設定を一覧し、その説明を記載しています。

