---
title: アーティファクト データのプライバシーとコンプライアンス
description: W&B ファイルがデフォルトでどこに保存されるかを確認できます。機密情報の保存方法についても解説します。
menu:
  default:
    identifier: data-privacy-and-compliance
    parent: artifacts
---

ファイルは、Artifacts をログする際に W&B が管理する Google Cloud のバケットにアップロードされます。バケット内のコンテンツは、保存時と転送時の両方で暗号化されます。Artifact ファイルは、対応する Project へのアクセス権を持つ Users だけが閲覧できます。

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B Client Server diagram" >}}

Artifact のバージョンを削除すると、その情報はデータベース上でソフト削除としてマークされ、ストレージコストから除外されます。Artifact 全体を削除した場合は、完全削除用のキューに入れられ、その内容が W&B バケットからすべて削除されます。ファイル削除に関して特別なご要望がある場合は、[Customer Support](mailto:support@wandb.com) までご連絡ください。

マルチテナント環境では管理できない機密性の高い Datasets については、クラウドバケットと接続したプライベートな W&B server や _reference artifacts_ を利用できます。Reference artifacts は、W&B にファイル内容を送信せずにプライベートバケットの参照だけを追跡します。Reference artifacts は、バケットやサーバー上のファイルへのリンクのみを保持します。つまり、W&B はファイル自体ではなく、ファイルに関連付けられたメタデータのみを管理します。

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B Client Server Cloud diagram" >}}

Reference artifact は通常の artifact とほぼ同じ手順で作成します。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

他の選択肢については、プライベートクラウドやオンプレミス環境の導入を含め、[contact@wandb.com](mailto:contact@wandb.com) までご相談ください。