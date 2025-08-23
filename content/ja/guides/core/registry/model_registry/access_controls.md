---
title: データガバナンスとアクセス制御を管理する
description: モデルレジストリのロールベース アクセス制御（RBAC）を使用して、保護されたエイリアスを更新できるユーザーを管理します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-access_controls
    parent: model-registry
weight: 10
---

*保護されたエイリアス* を使って、モデル開発パイプラインの主要なステージを表現しましょう。*Model Registry Administrators* のみが、保護されたエイリアスの追加・編集・削除を行うことができます。モデルレジストリ管理者は、保護されたエイリアスを定義・利用できます。W&B は管理者以外のユーザーによるモデルバージョンへの保護されたエイリアスの追加や削除をブロックします。

{{% alert %}}
Team 管理者または現在のレジストリ管理者のみが、レジストリ管理者のリストを管理できます。
{{% /alert %}}

例えば、`staging` や `production` を保護されたエイリアスとして設定した場合、チームのメンバーは誰でも新しいモデルバージョンを追加できます。しかし、`staging` や `production` エイリアスの追加は管理者のみが可能です。


## アクセス制御の設定方法
以下の手順で、チームのモデルレジストリに対するアクセス制御を設定します。

1. [W&B Model Registry アプリ](https://wandb.ai/registry/model)にアクセスします。
2. ページ右上にある歯車ボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="レジストリ設定の歯車" >}}
3. **Manage registry admins** ボタンを選択します。
4. **Members** タブで、モデルバージョンの保護されたエイリアスの追加や削除権限を与えたいユーザーを選択します。
{{< img src="/images/models/access_controls_admins.gif" alt="レジストリ管理者の管理" >}}


## 保護されたエイリアスの追加方法
1. [W&B Model Registry アプリ](https://wandb.ai/registry/model)にアクセスします。
2. ページ右上にある歯車ボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="レジストリ設定の歯車ボタン" >}}
3. **Protected Aliases** セクションまでスクロールします。
4. プラスアイコン（**+**）をクリックして、新しいエイリアスを追加します。
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="保護されたエイリアスの追加" >}}