---
title: How does W&B Launch build images?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_build_images
    parent: launch-faq
---

画像をビルドする手順は、ジョブのソースとリソース設定で指定されたアクセラレータベースイメージに依存します。

{{% alert %}}
キューを設定する際やジョブを提出する際には、キューまたはジョブのリソース設定にベースアクセラレータイメージを含めてください:
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
{{% /alert %}}

ビルド プロセスには、ジョブタイプと提供されたアクセラレータベースイメージに基づいた次の操作が含まれます。

| | apt を使用して Python をインストール | Python パッケージをインストール | ユーザーと作業ディレクトリを作成 | 画像にコードをコピー | エントリーポイントを設定 | |