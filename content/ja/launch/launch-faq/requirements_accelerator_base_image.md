---
title: アクセラレータのベースイメージにはどのような要件がありますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-requirements_accelerator_base_image
    parent: launch-faq
---

アクセラレータを利用するジョブでは、必要なアクセラレータコンポーネントを含むベースイメージを用意してください。アクセラレータイメージについて、以下の要件を満たしていることを確認してください:

- Debian と互換性があること（Launch Dockerfile では Python をインストールするために apt-get を使用します）
- CPU と GPU のハードウェア命令セットがサポートされていること（想定する GPU に対して CUDA バージョンの互換性を確認してください）
- 提供するアクセラレータのバージョンと機械学習アルゴリズム内のパッケージとの互換性
- ハードウェア互換性のために追加の手順を要するパッケージのインストール