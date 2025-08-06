---
title: カスタムレジストリを作成する
menu:
  default:
    identifier: create_registry
    parent: registry
weight: 2
---

カスタムレジストリは、使用できる artifact タイプの柔軟な管理や選択、レジストリの公開範囲の制限など、より高い柔軟性とコントロールを提供します。

{{% pageinfo color="info" %}}
コアレジストリとカスタムレジストリの比較については、[Registry types]({{< relref "registry_types.md#summary" >}}) のまとめ表をご覧ください。
{{% /pageinfo %}}


## カスタムレジストリの作成

カスタムレジストリを作成するには:

1. https://wandb.ai/registry/ の **Registry** アプリに移動します。
2. **Custom registry** セクションで、**Create registry** ボタンをクリックします。
3. **Name** フィールドに、レジストリの名前を入力します。
4. 必要に応じてレジストリの説明を追加します。
5. **Registry visibility** のドロップダウンから、レジストリを表示できるユーザーを選択します。レジストリの公開範囲オプションについて詳しくは [Registry visibility types]({{< relref "./configure_registry.md#registry-visibility-types" >}}) をご覧ください。
6. **Accepted artifacts type** のドロップダウンで **All types** または **Specify types** のいずれかを選択します。
7. （**Specify types** を選択した場合）レジストリで受け入れる artifact タイプを1つ以上追加します。
8. **Create registry** ボタンをクリックします。

{{% alert %}}
一度レジストリに保存された artifact タイプは、レジストリの設定から削除できません。
{{% /alert %}}

例えば、下図のように、ユーザーがまもなく作成しようとしている `Fine_Tuned_Models` というカスタムレジストリがあります。このレジストリは **Restricted**（制限付き）となっており、手動で追加されたメンバーだけが利用できます。

{{< img src="/images/registry/create_registry.gif" alt="Creating a new registry" >}}

## 公開範囲の種類（Visibility types）

レジストリの *公開範囲* は、そのレジストリへ誰がアクセスできるかを決定します。カスタムレジストリの公開範囲を制限することで、特定のメンバーのみがレジストリにアクセスできるようにできます。

カスタムレジストリでは、次の2種類の公開範囲オプションがあります。

| 公開範囲 | 説明 |
| --- | --- |
| Restricted   | 招待された組織メンバーのみが、レジストリへアクセス可能です。|
| Organization | 組織内のすべてのメンバーがレジストリへアクセス可能です。|

チーム管理者またはレジストリ管理者が、カスタムレジストリの公開範囲を設定できます。

Restricted（制限付き）の公開範囲でカスタムレジストリを作成した場合、その作成者は自動的にレジストリの管理者として追加されます。


## カスタムレジストリの公開範囲を設定する

チーム管理者またはレジストリ管理者は、カスタムレジストリの作成時または作成後に、公開範囲を設定できます。

既存のカスタムレジストリの公開範囲を制限する手順:

1. https://wandb.ai/registry/ の **Registry** アプリに移動します。
2. 公開範囲を変更したいレジストリを選択します。
3. 右上にある歯車アイコンをクリックします。
4. **Registry visibility** のドロップダウンから、希望する公開範囲を選択します。
5. **Restricted visibility** を選択した場合:
   1. このレジストリへのアクセスを許可したい組織のメンバーを追加します。**Registry members and roles** セクションで **Add member** ボタンをクリックしてください。
   2. **Member** フィールドに追加したいメンバーのメールアドレスまたはユーザー名を入力します。
   3. **Add new member** をクリックします。

{{< img src="/images/registry/change_registry_visibility.gif" alt="Changing registry visibility settings from private to public or team-restricted access" >}}

チーム管理者がカスタムレジストリを作成する際の公開範囲設定については、[カスタムレジストリを作成する]({{< relref "./create_registry.md#create-a-custom-registry" >}}) もご参照ください。