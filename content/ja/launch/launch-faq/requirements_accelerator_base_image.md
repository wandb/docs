---
title: アクセラレータのベースイメージにはどのような要件がありますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-requirements_accelerator_base_image
    parent: launch-faq
---

アクセラレータを使用するジョブの場合、必要なアクセラレータ コンポーネントを含むベースイメージを指定してください。アクセラレータ イメージには、以下の要件を満たしていることを確認してください。

- Debian との互換性（ Launch の Dockerfile は Python をインストールする際に apt-get を使用します）
- サポートされている CPU および GPU のハードウェア命令セット（使用予定の GPU に対応した CUDA バージョンであることを確認してください）
- 指定したアクセラレータ バージョンと、機械学習アルゴリズムに含まれるパッケージとの互換性
- ハードウェア互換性のために追加手順が必要なパッケージのインストール