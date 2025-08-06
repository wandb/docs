---
title: レジストリの種類
menu:
  default:
    identifier: registry_types
    parent: registry
weight: 1
---

W&B では 2 種類のレジストリがサポートされています: [Core registries]({{< relref "#core-registry" >}}) と [Custom registries]({{< relref "#custom-registry" >}}) です。

## Core registry
Core registry は特定のユースケースのためのテンプレートで、**Models** と **Datasets** があります。

デフォルトでは、**Models** レジストリは `"model"` アーティファクトタイプを受け付け、**Dataset** レジストリは `"dataset"` アーティファクトタイプを受け付けるように設定されています。管理者は受け入れ可能なアーティファクトタイプを追加できます。

{{< img src="/images/registry/core_registry_example.png" alt="Core registry" >}}

上の画像は、W&B Registry App UI での **Models** と **Dataset** のコアレジストリ、およびカスタムレジストリ **Fine_Tuned_Models** の例を示しています。

Core registry は [組織の公開範囲]({{< relref "./configure_registry.md#registry-visibility-types" >}}) となっています。レジストリ管理者はコアレジストリの公開範囲は変更できません。

## Custom registry
Custom registries は `"model"` や `"dataset"` のアーティファクトタイプに制限されません。

機械学習パイプラインの各ステップごとにカスタムレジストリを作成できます。たとえば、最初のデータ収集から最終的なモデルのデプロイメントまで柔軟に対応できます。

例として、トレーニング済みモデルの性能評価のためにキュレーション済みのデータセットを管理する "Benchmark_Datasets" というレジストリを作成できます。このレジストリ内に "User_Query_Insurance_Answer_Test_Data" というコレクションを作成し、そこにモデルがトレーニング中に見たことのないユーザーの質問と専門家が検証した回答セットを格納することができます。

{{< img src="/images/registry/custom_registry_example.png" alt="Custom registry example" >}}

Custom registry には [組織 または 制限付きの公開範囲]({{< relref "./configure_registry.md#registry-visibility-types" >}}) を設定できます。レジストリ管理者はカスタムレジストリの公開範囲を「組織」から「制限付き」へ変更することができます。ただし、「制限付き」から「組織」への変更はできません。

カスタムレジストリの作成方法については、[Create a custom registry]({{< relref "./create_collection.md" >}}) を参照してください。

## まとめ
以下の表は、Core registry と Custom registry の違いをまとめたものです。

|                | Core  | Custom|
| -------------- | ----- | ----- |
| 公開範囲       | 組織の公開範囲のみ。公開範囲は変更できません。 | 組織または制限付き。組織から制限付きには変更可能です。|
| メタデータ       | 事前設定されており、ユーザーによる編集はできません。 | ユーザーが編集可能。  |
| アーティファクトタイプ | 事前設定済みで、受け入れ可能なタイプは削除できません。追加することは可能です。 | 管理者が受け入れタイプを定義できます。|
| カスタマイズ    | 既存のリストにタイプを追加できます。| レジストリ名、説明、公開範囲、受け入れアーティファクトタイプを編集可能。|