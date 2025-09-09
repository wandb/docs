---
title: レジストリの種類
menu:
  default:
    identifier: ja-guides-core-registry-registry_types
    parent: registry
weight: 1
---

W&B は 2 種類のレジストリをサポートしています：[コアレジストリ]({{< relref path="#core-registry" lang="ja" >}}) と [カスタムレジストリ]({{< relref path="#custom-registry" lang="ja" >}})。

## コアレジストリ
コアレジストリは、特定のユースケース向けのテンプレートです：**Models** と **Datasets**。

デフォルトでは、**Models** レジストリは `"model"` Artifact タイプを受け入れるように、**Datasets** レジストリは `"dataset"` Artifact タイプを受け入れるように設定されています。管理者は、受け入れ可能な Artifact タイプを追加できます。

{{< img src="/images/registry/core_registry_example.png" alt="コアレジストリ" >}}

上の画像は、W&B Registry App UI における **Models** コアレジストリと **Datasets** コアレジストリ、および **Fine_Tuned_Models** というカスタムレジストリを示しています。

コアレジストリは [組織の公開範囲]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) を持ちます。レジストリ管理者はコアレジストリの公開範囲を変更できません。

## カスタムレジストリ
カスタムレジストリは、`"model"` Artifact タイプまたは `"dataset"` Artifact タイプに限定されません。

機械学習パイプラインの各ステップ（初期のデータ収集から最終的なモデルのデプロイメントまで）に対して、カスタムレジストリを作成できます。

例えば、「Benchmark_Datasets」というレジストリを作成し、トレーニング済みモデルのパフォーマンスを評価するために厳選されたデータセットを整理することができます。このレジストリ内には、「User_Query_Insurance_Answer_Test_Data」というコレクションがあるかもしれません。これは、モデルがトレーニング中に一度も見たことのない、ユーザーの質問とそれに対応する専門家によって検証された回答のセットを含んでいます。

{{< img src="/images/registry/custom_registry_example.png" alt="カスタムレジストリの例" >}}

カスタムレジストリは、[組織または制限された公開範囲]({{< relref path="./configure_registry.md#registry-visibility-types" lang="ja" >}}) のいずれかを持つことができます。レジストリ管理者はカスタムレジストリの公開範囲を組織から制限付きに変更できます。ただし、レジストリ管理者はカスタムレジストリの公開範囲を制限付きから組織の公開範囲に変更することはできません。

カスタムレジストリの作成方法については、[カスタムレジストリの作成]({{< relref path="./create_collection.md" lang="ja" >}}) を参照してください。

## 概要
以下の表は、コアレジストリとカスタムレジストリの違いをまとめたものです。

|                | Core                                                                | Custom                                                              |
| -------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 公開範囲       | 組織の公開範囲のみ。公開範囲は変更できません。                   | 組織または制限付きのいずれか。公開範囲は組織から制限付きに変更できます。 |
| メタデータ     | 事前設定されており、ユーザーは編集できません。                      | ユーザーが編集できます。                                           |
| Artifact タイプ | 事前設定されており、受け入れられる Artifact タイプは削除できません。ユーザーは、受け入れ可能な Artifact タイプを追加できます。 | 管理者が受け入れ可能な Artifact タイプを定義できます。             |
| カスタマイズ   | 既存のリストにタイプを追加できます。                               | レジストリ名、説明、公開範囲、および受け入れ可能な Artifact タイプを編集できます。 |