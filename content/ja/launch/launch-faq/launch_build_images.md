---
title: W&B Launch はどのようにしてイメージを作成しますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_build_images
    parent: launch-faq
---

画像のビルド手順は、ジョブのソースとリソース設定で指定されたアクセラレータのベース画像によって異なります。

{{% alert %}}
キューを設定する場合やジョブを送信する際には、キューやジョブのリソース設定にベースアクセラレータ画像を含めてください:
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

ビルドプロセスには、ジョブタイプと提供されたアクセラレータのベース画像に基づいて、以下のアクションが含まれます:

| | apt を使用して Python をインストール | Python パッケージをインストール | ユーザーと作業ディレクトリを作成 | コードを画像にコピー | エントリーポイントを設定 | |
