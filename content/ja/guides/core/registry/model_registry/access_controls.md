---
title: Manage data governance and access control
description: モデルレジストリのロールベース アクセス制御（RBAC）を使用して、保護されたエイリアスを更新できるユーザーを制御します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-access_controls
    parent: model-registry
weight: 10
---

モデル開発 パイプライン の主要な段階を表すために、 *保護されたエイリアス* を使用します。 *モデルレジストリ管理者* のみが、保護されたエイリアスを追加、変更、または削除できます。モデルレジストリ管理者は、保護されたエイリアスを定義して使用できます。W&B は、管理者以外の ユーザー がモデル バージョン から保護されたエイリアスを追加または削除することをブロックします。

{{% alert %}}
Team 管理者または現在のレジストリ管理者のみが、レジストリ管理者のリストを管理できます。
{{% /alert %}}

たとえば、 `staging` と `production` を保護されたエイリアスとして設定するとします。Team のメンバーは誰でも新しいモデル バージョンを追加できます。ただし、管理者のみが `staging` または `production` エイリアスを追加できます。

## アクセス制御の設定

次の手順では、Team の モデルレジストリ の アクセス 制御を設定する方法について説明します。

1. W&B モデルレジストリ アプリ ( [https://wandb.ai/registry/model](https://wandb.ai/registry/model) ) に移動します。
2. ページの右上にある歯車ボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="" >}}
3. [**レジストリ管理者の管理**] ボタンを選択します。
4. [**メンバー**] タブで、モデル バージョン から保護されたエイリアスを追加および削除する アクセス 権を付与する ユーザー を選択します。
{{< img src="/images/models/access_controls_admins.gif" alt="" >}}

## 保護されたエイリアスの追加

1. W&B モデルレジストリ アプリ ( [https://wandb.ai/registry/model](https://wandb.ai/registry/model) ) に移動します。
2. ページの右上にある歯車ボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="" >}}
3. [**保護されたエイリアス**] セクションまで下にスクロールします。
4. プラス アイコン ( **+** ) アイコンをクリックして、新しいエイリアスを追加します。
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="" >}}
