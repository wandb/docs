---
title: カスタムレジストリを作成する
menu:
  default:
    identifier: ja-guides-core-registry-create_registry
    parent: registry
weight: 2
---

カスタムレジストリを使うことで、利用可能な artifact の種類を柔軟に制御でき、レジストリの公開範囲の制限なども行えます。

{{% pageinfo color="info" %}}
コアレジストリとカスタムレジストリの違いについては、[レジストリタイプのまとめ]({{< relref path="registry_types.md#summary" lang="ja" >}})の表をご覧ください。
{{% /pageinfo %}}


## カスタムレジストリの作成

カスタムレジストリを作成するには:

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
2. **Custom registry** 内で、**Create registry** ボタンをクリックします。
3. **Name** フィールドに、レジストリの名前を入力します。
4. 必要に応じて、レジストリの説明を入力します。
5. **Registry visibility** ドロップダウンから、レジストリを閲覧できるユーザーを選びます。レジストリの公開範囲については[レジストリ公開範囲の種類]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}})をご参照ください。
6. **Accepted artifacts type** ドロップダウンから、**All types** または **Specify types** を選択します。
7. （**Specify types** を選択した場合）レジストリで受け入れる artifact の種類を1つ以上追加します。
8. **Create registry** ボタンをクリックします。

{{% alert %}}
artifact の種類は、一度レジストリの設定に保存すると後から削除できません。
{{% /alert %}}

例えば、次の画像は、`Fine_Tuned_Models` というカスタムレジストリを作成しようとしている例です。このレジストリは**Restricted**（制限付き）で、手動で追加されたメンバーだけが利用できます。

{{< img src="/images/registry/create_registry.gif" alt="Creating a new registry" >}}

## 公開範囲の種類

レジストリの *公開範囲* は、そのレジストリに誰がアクセスできるかを決定します。公開範囲を制限することで、特定のメンバーのみがアクセスできるようにできます。

カスタムレジストリの公開範囲オプションは2つあります:

| 公開範囲 | 説明 |
| --- | --- | 
| Restricted（制限付き）   | 招待された組織メンバーのみがレジストリにアクセス可能です。| 
| Organization（組織全体） | 組織内の全員がレジストリにアクセスできます。 |

チーム管理者またはレジストリ管理者が、カスタムレジストリの公開範囲を設定できます。

**Restricted** な公開範囲でカスタムレジストリを作成した場合、作成者は自動的にそのレジストリの管理者として追加されます。


## カスタムレジストリの公開範囲を設定する

チーム管理者またはレジストリ管理者は、カスタムレジストリの作成時や作成後に、その公開範囲を指定できます。

既存のカスタムレジストリの公開範囲を制限するには:

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
2. 対象のレジストリを選択します。
3. 右上の歯車アイコンをクリックします。
4. **Registry visibility** ドロップダウンから、希望する公開範囲を選択します。
5. **Restricted visibility** を選択した場合は、以下の手順に従います:
   1. レジストリにアクセスさせたい組織メンバーを追加します。**Registry members and roles** セクションまでスクロールし、**Add member** ボタンをクリックします。
   2. **Member** フィールドに、追加するメンバーのメールアドレスまたはユーザー名を入力します。
   3. **Add new member** をクリックします。

{{< img src="/images/registry/change_registry_visibility.gif" alt="Changing registry visibility settings from private to public or team-restricted access" >}}

チーム管理者がカスタムレジストリを作成する際の公開範囲の設定方法については、[カスタムレジストリの作成]({{< relref path="./create_registry.md#create-a-custom-registry" lang="ja" >}})をご覧ください。