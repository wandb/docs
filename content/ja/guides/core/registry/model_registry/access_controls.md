---
title: データ ガバナンスとアクセス コントロールを管理する
description: モデルレジストリのロールベース アクセス制御 (RBAC) を使用して、誰が保護されたエイリアスを更新できるかを制御します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-access_controls
    parent: model-registry
weight: 10
---

*保護されたエイリアス* を使用して、モデル開発パイプラインの主要なステージを表現します。*モデルレジストリ管理者* のみが保護されたエイリアスを追加、変更、または削除できます。モデルレジストリ管理者は保護されたエイリアスを定義し、使用することができます。W&B は非管理ユーザーがモデルバージョンから保護されたエイリアスを追加または削除することをブロックします。

{{% alert %}}
チーム管理者または現在のレジストリ管理者のみがレジストリ管理者のリストを管理できます。
{{% /alert %}}

例えば、`staging` と `production` を保護されたエイリアスとして設定したとします。チームのどのメンバーも新しいモデルバージョンを追加できますが、`staging` または `production` エイリアスを追加できるのは管理者のみです。

## アクセス制御の設定

次の手順で、チームのモデルレジストリに対するアクセス制御を設定します。

1. W&B モデルレジストリアプリケーションに移動します：[https://wandb.ai/registry/model](https://wandb.ai/registry/model)
2. ページ右上のギアボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="" >}}
3. **Manage registry admins** ボタンを選択します。
4. **Members** タブ内で、モデルバージョンから保護されたエイリアスを追加および削除するアクセス権を付与したいユーザーを選択します。
{{< img src="/images/models/access_controls_admins.gif" alt="" >}}

## 保護されたエイリアスの追加

1. W&B モデルレジストリアプリケーションに移動します：[https://wandb.ai/registry/model](https://wandb.ai/registry/model)
2. ページ右上のギアボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="" >}}
3. **Protected Aliases** セクションまでスクロールダウンします。
4. プラスアイコン (**+**) をクリックして新しいエイリアスを追加します。
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="" >}}