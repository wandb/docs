---
title: データガバナンスとアクセス制御を管理する
description: モデルレジストリのロールベースアクセス制御（RBAC）を使用して、保護されたエイリアスを更新できるユーザーを管理します。
menu:
  default:
    identifier: access_controls
    parent: model-registry
weight: 10
---

*protected aliases* を使って、モデル開発パイプラインの重要なステージを表現しましょう。*Model Registry Administrators* （モデルレジストリ管理者）のみが、protected aliases の追加・変更・削除を行うことができます。モデルレジストリの管理者だけが protected alias の設定・利用を行うことができます。W&B では管理者以外のユーザーが model version から protected aliases を追加・削除することをブロックします。

{{% alert %}}
Team 管理者、または現在の registry 管理者のみが registry 管理者リストを管理できます。
{{% /alert %}}

例えば、`staging` および `production` を protected aliases として設定した場合、チームのどのメンバーも新しい model version を追加できます。しかし、管理者のみが `staging` または `production` エイリアスを追加できます。


## アクセス制御の設定方法
以下のステップでは、チームのモデルレジストリに対してアクセス制御を設定する方法を説明します。

1. [W&B Model Registry アプリ](https://wandb.ai/registry/model)にアクセスします。
2. ページ右上のギアボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="Registry 設定ギア" >}}
3. **Manage registry admins** ボタンを選択します。
4. **Members** タブ内で、model version への protected aliases の追加・削除権限を与えたいユーザーを選択します。
{{< img src="/images/models/access_controls_admins.gif" alt="Registry 管理者の管理" >}}


## Protected aliases の追加方法
1. [W&B Model Registry アプリ](https://wandb.ai/registry/model)にアクセスします。
2. ページ右上のギアボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="Registry 設定ギアボタン" >}}
3. **Protected Aliases** セクションまでスクロールします。
4. プラスアイコン（**+**）をクリックして、新しい alias を追加します。
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="Protected aliases の追加" >}}