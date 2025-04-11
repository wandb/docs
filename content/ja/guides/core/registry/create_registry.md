---
title: カスタムレジストリを作成する
menu:
  default:
    identifier: ja-guides-core-registry-create_registry
    parent: registry
weight: 2
---

カスタムレジストリは、使用できるアーティファクトタイプに関して柔軟性とコントロールを提供し、レジストリの公開範囲を制限することができるなどの機能があります。

{{% pageinfo color="info" %}}
コアとカスタムレジストリの完全な比較は、[Registry types]({{< relref path="registry_types.md#summary" lang="ja" >}})の概要表をご覧ください。
{{% /pageinfo %}}

## カスタムレジストリを作成する

カスタムレジストリを作成するには:
1. https://wandb.ai/registry/ の **Registry** アプリに移動します。
2. **Custom registry** 内で、**Create registry** ボタンをクリックします。
3. **Name** フィールドにレジストリの名前を入力します。
4. 必要に応じてレジストリの説明を提供します。
5. **Registry visibility** ドロップダウンからレジストリを閲覧できる人を選択します。レジストリの公開範囲オプションの詳細については、[Registry visibility types]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}})をご覧ください。
6. **All types** または **Specify types** を **Accepted artifacts type** ドロップダウンから選択します。
7. （**Specify types** を選択した場合）レジストリが受け入れる1つ以上のアーティファクトタイプを追加します。
8. **Create registry** ボタンをクリックします。

{{% alert %}}
アーティファクトタイプは、一旦レジストリの設定に保存されるとそのレジストリから削除することはできません。
{{% /alert %}}

たとえば、以下の画像はユーザーが作成しようとしている `Fine_Tuned_Models` というカスタムレジストリを示しています。このレジストリは、手動でレジストリに追加されたメンバーのみに**制限**されています。

{{< img src="/images/registry/create_registry.gif" alt="" >}}

## 公開範囲タイプ

レジストリの*公開範囲*は、誰がそのレジストリにアクセスできるかを決定します。カスタムレジストリの公開範囲を制限すると、指定されたメンバーのみがそのレジストリにアクセスできるようにするのに役立ちます。

カスタムレジストリには2つの公開範囲オプションがあります:

| 公開範囲 | 説明 |
| --- | --- | 
| Restricted   | 招待された組織メンバーのみがレジストリにアクセスできます。| 
| Organization | 組織内の全員がレジストリにアクセスできます。|

チーム管理者またはレジストリ管理者は、カスタムレジストリの公開範囲を設定できます。

Restricted公開範囲でカスタムレジストリを作成したユーザーは、自動的にそのレジストリの管理者として登録されます。

## カスタムレジストリの公開範囲を設定する

チーム管理者またはレジストリ管理者は、カスタムレジストリの作成時または作成後に公開範囲を設定することができます。

既存のカスタムレジストリの公開範囲を制限するには:

1. https://wandb.ai/registry/ の **Registry** アプリに移動します。
2. 任意のレジストリを選択します。
3. 右上隅の歯車アイコンをクリックします。
4. **Registry visibility** ドロップダウンから、希望するレジストリの公開範囲を選択します。
5. **Restricted visibility** を選択した場合:
   1. このレジストリにアクセスを許可したい組織のメンバーを追加します。 **Registry members and roles** セクションまでスクロールし、**Add member** ボタンをクリックします。
   2. **Member** フィールドに追加したいメンバーのメールまたはユーザー名を入力します。
   3. **Add new member** をクリックします。

{{< img src="/images/registry/change_registry_visibility.gif" alt="" >}}

チーム管理者がそれを作成する際に、カスタムレジストリの公開範囲をどのように設定するかに関する詳細は [Create a custom registry]({{< relref path="./create_registry.md#create-a-custom-registry" lang="ja" >}}) を参照してください。