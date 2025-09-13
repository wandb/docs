---
title: データガバナンスとアクセス制御を管理する
description: Model Registry のロールベースのアクセス制御 (RBAC) を使用して、保護されたエイリアスを更新できるユーザーを管理します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-access_controls
    parent: model-registry
weight: 10
---

モデル開発パイプラインの主要な段階を表すには、*保護されたエイリアス* を使います。*モデルレジストリ管理者* だけが保護されたエイリアスを追加・変更・削除できます。モデルレジストリ管理者は、保護されたエイリアスを定義して利用できます。W&B は、管理者以外のユーザーがモデルのバージョンに対して保護されたエイリアスを追加または削除することをブロックします。
{{% alert %}}
チーム管理者または現在のレジストリ管理者だけが、レジストリ管理者のリストを管理できます。
{{% /alert %}}
たとえば、`staging` と `production` を保護されたエイリアスとして設定したとします。チームのメンバーであれば誰でも新しいモデルのバージョンを追加できますが、`staging` や `production` のエイリアスを追加できるのは管理者だけです。
## アクセスコントロールの設定
以下の手順は、チームのモデルレジストリのアクセスコントロールの設定方法を説明します。
1. [W&B モデルレジストリ アプリ](https://wandb.ai/registry/model) に移動します。
2. ページの右上にある歯車ボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="レジストリ設定の歯車" >}}
3. **レジストリ管理者の管理** ボタンを選択します。
4. **メンバー** タブ内で、モデルのバージョンから保護されたエイリアスを追加および削除するアクセスを付与したいユーザーを選択します。
{{< img src="/images/models/access_controls_admins.gif" alt="レジストリ管理者の管理" >}}
## 保護されたエイリアスの追加
1. [W&B モデルレジストリ アプリ](https://wandb.ai/registry/model) に移動します。
2. ページの右上にある歯車ボタンを選択します。
{{< img src="/images/models/rbac_gear_button.png" alt="レジストリ設定の歯車ボタン" >}}
3. **保護されたエイリアス** セクションまでスクロールします。
4. プラスアイコン (**+**) をクリックして、新しいエイリアスを追加します。
{{< img src="/images/models/access_controls_add_protected_aliases.gif" alt="保護されたエイリアスの追加" >}}