---
description: モデルレジストリのロールベースアクセス制御 (RBAC) を使用して、保護されたエイリアスを更新できるユーザーを制御します。
displayed_sidebar: default
---


# データガバナンスとアクセス制御

*プロテクトされたエイリアス* を使用して、モデル開発パイプラインの重要なステージを表現します。*モデルレジストリ管理者* のみがプロテクトされたエイリアスを追加、変更、削除できます。モデルレジストリの管理者はプロテクトされたエイリアスを定義および使用することができます。W&B は管理者以外のユーザーがモデルバージョンからプロテクトされたエイリアスを追加または削除することをブロックします。

:::info
チーム管理者または現在のレジストリ管理者のみが、レジストリ管理者のリストを管理できます。
:::

例えば、`staging` と `production` をプロテクトされたエイリアスとして設定したとします。チームの任意のメンバーは新しいモデルバージョンを追加できます。しかし、管理者のみが `staging` や `production` エイリアスを追加できます。

## アクセス制御の設定
次の手順は、チームのモデルレジストリに対するアクセス制御の設定方法を説明しています。

1. W&B Model Registry アプリにアクセスします：[https://wandb.ai/registry/model](https://wandb.ai/registry/model)
2. ページの右上にあるギアボタンを選択します。
![](/images/models/rbac_gear_button.png)
3. **Manage registry admins** ボタンを選択します。
4. **Members** タブ内で、モデルバージョンからプロテクトされたエイリアスを追加および削除する権限を与えたいユーザーを選択します。
![](/images/models/access_controls_admins.gif)

## プロテクトされたエイリアスの追加
1. W&B Model Registry アプリにアクセスします：[https://wandb.ai/registry/model](https://wandb.ai/registry/model)
2. ページの右上にあるギアボタンを選択します。
![](/images/models/rbac_gear_button.png)
3. **Protected Aliases** セクションまでスクロールダウンします。
4. プラスアイコン (**+**) をクリックして、新しいエイリアスを追加します。
![](/images/models/access_controls_add_protected_aliases.gif)