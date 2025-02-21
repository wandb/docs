---
title: Registry types
menu:
  default:
    identifier: ja-guides-models-registry-registry_types
    parent: registry
weight: 1
---

W&B は、[コア registry]({{< relref path="#core-registry" lang="ja" >}}) と [カスタム registry]({{< relref path="#custom-registry" lang="ja" >}}) の 2 種類の registry をサポートしています。

## コア registry
コア registry は、特定のユースケース（**Models** と **Datasets**）のテンプレートです。

デフォルトでは、**Models** registry は `"model"` artifact タイプを受け入れるように構成されており、**Dataset** registry は `"dataset"` artifact タイプを受け入れるように構成されています。管理者は、追加で許可される artifact タイプを追加できます。

{{< img src="/images/registry/core_registry_example.png" alt="" >}}

上の画像は、W&B Registry App UI の **Models** と **Dataset** のコア registry と、**Fine_Tuned_Models** という名前のカスタム registry を示しています。

コア registry は、[組織の可視性]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) を持っています。 registry 管理者は、コア registry の可視性を変更できません。

## カスタム registry
カスタム registry は、`"model"` artifact タイプまたは `"dataset"` artifact タイプに限定されません。

最初のデータ収集から最終的な model のデプロイメントまで、機械学習 パイプライン の各ステップに対してカスタム registry を作成できます。

たとえば、トレーニングされた model のパフォーマンスを評価するために、キュレーションされたデータセットを整理するための "Benchmark_Datasets" という registry を作成できます。この registry 内には、トレーニング中に model が見たことのない一連のユーザーの質問と、対応する専門家によって検証された回答を含む "User_Query_Insurance_Answer_Test_Data" というコレクションがあるかもしれません。

{{< img src="/images/registry/custom_registry_example.png" alt="" >}}

カスタム registry は、[組織または制限付きの可視性]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) を持つことができます。 registry 管理者は、カスタム registry の可視性を組織から制限付きに変更できます。ただし、registry 管理者は、カスタム registry の可視性を制限付きから組織の可視性に変更することはできません。

カスタム registry の作成方法については、[カスタム registry を作成する]({{< relref path="./create_collection.md" lang="ja" >}}) を参照してください。

## まとめ
次の表は、コア registry とカスタム registry の違いをまとめたものです。

|                | Core  | Custom|
| -------------- | ----- | ----- |
| 可視性     | 組織の可視性のみ。可視性は変更できません。 | 組織または制限付き。可視性は組織から制限付きの可視性に変更できます。|
| メタデータ       | 事前に構成されており、ユーザーは編集できません。 | ユーザーは編集できます。  |
| Artifact タイプ | 事前に構成されており、承認された artifact タイプを削除することはできません。ユーザーは、追加で許可される artifact タイプを追加できます。 | 管理者は、許可されるタイプを定義できます。 |
| カスタマイズ    | 既存のリストに追加のタイプを追加できます。|  registry 名、説明、可視性、および許可される artifact タイプを編集します。|
