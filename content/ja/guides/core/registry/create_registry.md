---
title: カスタムレジストリを作成する
menu:
  default:
    identifier: ja-guides-core-registry-create_registry
    parent: registry
weight: 2
---

カスタムレジストリは、使用できる Artifacts タイプに対する柔軟性と制御を提供し、レジストリの公開範囲を制限することなどを可能にします。

{{% pageinfo color="info" %}}
コアレジストリとカスタムレジストリの完全な比較については、[レジストリタイプ]({{< relref path="registry_types.md#summary" lang="ja" >}})の概要表を参照してください。
{{% /pageinfo %}}

## カスタムレジストリを作成する

カスタムレジストリを作成するには：
1. https://wandb.ai/registry/ にある **Registry** アプリに移動します。
2. **Custom registry** 内で、**Create registry** ボタンをクリックします。
3. **Name** フィールドにレジストリの名前を指定します。
4. 必要に応じてレジストリに関する説明を指定します。
5. **Registry visibility** ドロップダウンから、レジストリを表示できるユーザーを選択します。レジストリの公開範囲オプションに関する詳細については、[レジストリの公開範囲のタイプ]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}})を参照してください。
6. **Accepted artifacts type** ドロップダウンから、**All types** または **Specify types** のいずれかを選択します。
7. (**Specify types** を選択した場合) レジストリが受け入れる 1 つ以上の Artifacts タイプを追加します。
8. **Create registry** ボタンをクリックします。

{{% alert %}}
レジストリの **設定** に保存された後は、Artifacts タイプをレジストリから削除することはできません。
{{% /alert %}}

例えば、次の画像は、ユーザーが作成しようとしている `Fine_Tuned_Models` というカスタムレジストリを示しています。このレジストリは、手動で追加されたメンバーのみに **Restricted** されています。

{{< img src="/images/registry/create_registry.gif" alt="新しいレジストリの作成" >}}

## 公開範囲のタイプ

レジストリの *公開範囲* は、そのレジストリにアクセスできるユーザーを決定します。カスタムレジストリの公開範囲を制限することで、指定されたメンバーのみがそのレジストリにアクセスできることを確実にすることができます。

カスタムレジストリには、2 種類のレジストリの公開範囲オプションがあります：

| 公開範囲 | 説明 |
| --- | --- |
| Restricted | 招待された組織メンバーのみがレジストリにアクセスできます。|
| Organization | 組織内のすべてのユーザーがレジストリにアクセスできます。 |

チーム管理者またはレジストリ管理者は、カスタムレジストリの公開範囲を設定できます。

Restricted の公開範囲を持つカスタムレジストリを作成するユーザーは、そのレジストリ管理者として自動的にレジストリに追加されます。

## カスタムレジストリの公開範囲を **設定** する

チーム管理者またはレジストリ管理者は、カスタムレジストリの作成中または作成後に、その公開範囲を割り当てることができます。

既存のカスタムレジストリの公開範囲を制限するには：

1. https://wandb.ai/registry/ にある **Registry** アプリに移動します。
2. レジストリを選択します。
3. 右上隅にある歯車アイコンをクリックします。
4. **Registry visibility** ドロップダウンから、目的のレジストリの公開範囲を選択します。
5. **Restricted visibility** を選択した場合：
   1. このレジストリにアクセスさせたい組織のメンバーを追加します。**Registry members and roles** セクションまでスクロールし、**Add member** ボタンをクリックします。
   2. **Member** フィールド内で、追加したいメンバーのメールアドレスまたはユーザー名を追加します。
   3. **Add new member** をクリックします。

{{< img src="/images/registry/change_registry_visibility.gif" alt="レジストリの公開範囲設定をプライベートから公開またはチーム限定アクセスに変更" >}}

チーム管理者がカスタムレジストリを作成する際に、その公開範囲を割り当てる方法に関する詳細については、[カスタムレジストリを作成する]({{< relref path="./create_registry.md#create-a-custom-registry" lang="ja" >}})を参照してください。