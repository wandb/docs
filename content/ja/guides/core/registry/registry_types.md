---
title: Registryタイプ
menu:
  default:
    identifier: ja-guides-core-registry-registry_types
    parent: registry
weight: 1
---

W&B では、2 種類のRegistryがサポートされています：[Core Registry]({{< relref path="#core-registry" lang="ja" >}}) と [Custom Registry]({{< relref path="#custom-registry" lang="ja" >}}) です。

## Core Registry
Core Registryは、特定のユースケース（**Models** と **Datasets** ）向けのテンプレートです。

デフォルトでは、**Models** Registryは `"model"` アーティファクトタイプを受け入れるように設定されており、**Dataset** Registryは `"dataset"` アーティファクトタイプを受け入れるように設定されています。管理者は追加で受け入れるアーティファクトタイプを追加できます。

{{< img src="/images/registry/core_registry_example.png" alt="Core registry" >}}

上の画像は、**Models** と **Dataset** の Core Registry、およびカスタムRegistryである **Fine_Tuned_Models** が W&B Registry App UI に表示されている様子です。

Core Registryには [組織公開範囲]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) があります。Registry管理者は Core Registryの公開範囲を変更できません。

## Custom Registry
Custom Registryは `"model"` や `"dataset"` といったアーティファクトタイプに制限されません。

初期のデータ収集から最終的なモデルのデプロイメントまで、機械学習パイプラインの各ステップごとに Custom Registryを作成できます。

例えば、「Benchmark_Datasets」というRegistryを作成し、トレーニング済みモデルの性能評価用にキュレーション済みのデータセットを整理できます。このRegistry内に「User_Query_Insurance_Answer_Test_Data」というコレクションを作成し、モデルがトレーニング中に見たことのないユーザーの質問と専門家によって検証された回答のセットを格納しておく、といった使い方も可能です。

{{< img src="/images/registry/custom_registry_example.png" alt="Custom registry example" >}}

Custom Registryの公開範囲は [組織または制限付き公開範囲]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) から選択できます。Registry管理者は Custom Registryの公開範囲を「組織」から「制限付き」へと変更できますが、「制限付き」から「組織」への変更はできません。

Custom Registryの作成方法については、[Custom Registryの作成]({{< relref path="./create_collection.md" lang="ja" >}}) をご覧ください。

## まとめ
下表は、Core Registryと Custom Registryの違いをまとめたものです。

|                | Core  | Custom|
| -------------- | ----- | ----- |
| 公開範囲     | 組織公開範囲のみ。公開範囲は変更できません。 | 組織または制限付きのいずれか。公開範囲は「組織」から「制限付き」へ変更可能。|
| メタデータ       | 事前設定されており、ユーザーは編集できません。 | ユーザーが編集可能。 |
| アーティファクトタイプ | 事前設定されており、受け入れられるタイプは削除できませんが、追加は可能。 | 管理者が受け入れるタイプを定義可能。 |
| カスタマイズ    | 既存リストへのタイプ追加が可能。|  Registry名、説明、公開範囲、受け入れるアーティファクトタイプを編集可能。|