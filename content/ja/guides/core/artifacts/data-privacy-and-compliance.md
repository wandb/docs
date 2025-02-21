---
title: Artifact data privacy and compliance
description: W&B のファイルがデフォルトでどこに保存されるかを確認します。機密情報を保存、保管する方法を説明します。
menu:
  default:
    identifier: ja-guides-core-artifacts-data-privacy-and-compliance
    parent: artifacts
---

ファイルを Artifacts として ログに記録すると、W&B が管理する Google Cloud バケットにアップロードされます。バケットの内容は、保存時および転送中に暗号化されます。Artifact のファイルは、対応する Projects へのアクセス権を持つ ユーザー のみに表示されます。

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B Client Server diagram" >}}

Artifact の バージョン を削除すると、データベースで論理削除としてマークされ、ストレージコストから削除されます。Artifact 全体を削除すると、完全に削除するためにキューに入れられ、そのすべてのコンテンツが W&B バケット から削除されます。ファイルの削除に関して特別なニーズがある場合は、[カスタマーサポート](mailto:support@wandb.com) までご連絡ください。

マルチテナント 環境 に存在できない機密性の高い データセット の場合、 クラウド バケット に接続された プライベート W&B サーバー または、_参照 Artifacts_ のいずれかを使用できます。参照 Artifacts は、ファイルの内容を W&B に送信せずに、プライベート バケット への参照を追跡します。参照 Artifacts は、 バケット または サーバー 上のファイルへのリンクを保持します。言い換えれば、W&B はファイル自体ではなく、ファイルに関連付けられた メタデータ のみを追跡します。

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B Client Server Cloud diagram" >}}

参照 Artifact を作成する方法は、非参照 Artifact を作成する方法と似ています。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

代替手段については、[contact@wandb.com](mailto:contact@wandb.com) までお問い合わせいただき、プライベート クラウド および オンプレミス のインストールについてご相談ください。
