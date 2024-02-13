---
description: >-
  Learn where W&B files are stored by default. Explore how to save, store
  sensitive information.
displayed_sidebar: ja
---

# データプライバシーとコンプライアンス

<head>
    <title>アーティファクトのデータプライバシーとコンプライアンス</title>
</head>
アーティファクトをログに記録する際、ファイルはWeights & Biasesが管理するGoogle Cloudバケットにアップロードされます。バケットの内容は、保存時も転送時も暗号化されています。アーティファクトファイルは、対応するプロジェクトへのアクセス権があるユーザーにのみ表示されます。

![GCS W&B クライアントサーバーダイアグラム](/images/artifacts/data_and_privacy_compliance_1.png)

アーティファクトのバージョンを削除すると、安全に削除できるファイル（前のバージョンや次のバージョンで使用されていないファイル）が、Weights & Biasesのバケットから _直ちに_ 削除されます。同様に、アーティファクト全体を削除すると、その内容全てがバケットから削除されます。

マルチテナント環境に配置できない機密データセットの場合、プライベートなW&Bサーバーをクラウドバケットに接続するか、_リファレンスアーティファクト_ を使用できます。リファレンスアーティファクトは、ファイルの内容をW&Bに送信せずに、プライベートバケットへの参照をトラッキングします。リファレンスアーティファクトは、バケットやサーバー上のファイルへのリンクを維持します。つまり、Weights & Biasesはファイルに関連するメタデータのみを追跡し、ファイル自体は保持しません。

![W&B クライアントサーバークラウド図](/images/artifacts/data_and_privacy_compliance_2.png)

参照用アーティファクトを、非参照用アーティファクトの作成方法と同様に作成してください：

```python
import wandb
```
run = wandb.init()
アーティファクト = wandb.Artifact('動物', type='データセット')
アーティファクト.add_reference('s3://my-バケット/動物')
```

代替案については、プライベートクラウドおよびオンプレミスのインストールについて話し合うために[contact@wandb.com](mailto:contact@wandb.com)までお問い合わせください。