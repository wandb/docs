---
title: Artifacts データのプライバシーとコンプライアンス
description: W&B のファイルがデフォルトでどこに保存されるかを確認し、シークレットの保存方法を学びます。
menu:
  default:
    identifier: ja-guides-core-artifacts-data-privacy-and-compliance
    parent: artifacts
---

Artifacts をログすると、W&B が管理する Google Cloud バケットにファイルがアップロードされます。バケット内のコンテンツは、保管時と転送時の両方で暗号化されます。Artifact のファイルは、対応する Project へのアクセス権を持つ Users のみが閲覧できます。

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B クライアントサーバー図" >}}

Artifact のバージョンを削除すると、データベース内で論理削除としてマークされ、ストレージコストから除外されます。Artifact 全体を削除すると、完全削除のキューに入れられ、そのすべてのコンテンツが W&B バケットから削除されます。ファイル削除に関して特定の要件がある場合は、[カスタマーサポート](mailto:support@wandb.com) にお問い合わせください。

マルチテナント環境に置くことができない機密性の高いデータセットの場合、クラウド バケットに接続されたプライベート W&B サーバー、または _参照 Artifacts_ を使用できます。参照 Artifacts は、ファイルのコンテンツを W&B に送信することなく、プライベート バケットへの参照を追跡します。参照 Artifacts は、ご自身のバケットまたはサーバー上のファイルへのリンクを維持します。つまり、W&B はファイル自体ではなく、ファイルに関連付けられたメタデータのみを追跡します。

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B クライアントサーバークラウド図" >}}

非参照の Artifact の作成方法と同様に、参照 Artifact を作成します。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

代替案については、[contact@wandb.com](mailto:contact@wandb.com) までお問い合わせください。プライベートクラウドおよびオンプレミスのインストールについてご相談いただけます。