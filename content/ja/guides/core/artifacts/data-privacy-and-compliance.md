---
title: Artifact data privacy and compliance
description: W&B のファイルがデフォルトでどこに保存されるかについて学びましょう。機密情報の保存、保存方法について説明します。
menu:
  default:
    identifier: ja-guides-core-artifacts-data-privacy-and-compliance
    parent: artifacts
---

ファイルを Artifacts としてログに記録すると、ファイルは W&B が管理する Google Cloud バケットにアップロードされます。バケットの内容は、保存時も転送時も暗号化されます。Artifact ファイルは、対応する プロジェクト への アクセス権 を持つ ユーザー のみに表示されます。

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B Client Server diagram" >}}

Artifact の バージョン を削除すると、 データベース 内でソフト削除としてマークされ、ストレージコストから削除されます。Artifact 全体を削除すると、 完全に削除されるようにキューに入れられ、そのすべてのコンテンツが W&B バケット から削除されます。ファイルの削除に関して特別なニーズがある場合は、[カスタマーサポート](mailto:support@wandb.com)までご連絡ください。

マルチテナント 環境 に存在できない機密性の高い データセット の場合、 クラウド バケット に接続された プライベート W&B サーバー 、または _reference artifacts_ を使用できます。Reference artifacts は、ファイルの内容を W&B に送信せずに、 プライベート バケット への参照を追跡します。Reference artifacts は、 バケット または サーバー 上のファイルへのリンクを保持します。つまり、W&B はファイル自体ではなく、ファイルに関連付けられた メタデータ のみを追跡します。

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B Client Server Cloud diagram" >}}

Reference artifact は、非 Reference artifact を作成するのと同じように作成します。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

代替案については、[contact@wandb.com](mailto:contact@wandb.com) までお問い合わせいただき、 プライベートクラウド および オンプレミス のインストールについてご相談ください。
