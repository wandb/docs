---
title: W&B Launch はどのようにイメージをビルドしますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_build_images
    parent: launch-faq
---

イメージのビルド手順は、ジョブのソースと、リソース設定で指定されたアクセラレータのベースイメージに依存します。

{{% alert %}}
キューを設定する際、またはジョブを送信する際は、キューまたはジョブのリソース設定にベースのアクセラレータイメージを含めてください:
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

ビルドプロセスには、ジョブタイプ と提供されたアクセラレータのベースイメージに基づき、次の処理が含まれます:

| | apt を使って Python をインストール | Python パッケージをインストール | ユーザー と作業ディレクトリを作成 | イメージに コード をコピー | エントリポイントを設定 | |