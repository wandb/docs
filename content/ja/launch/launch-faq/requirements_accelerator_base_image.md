---
title: What requirements does the accelerator base image have?
menu:
  launch:
    identifier: ja-launch-launch-faq-requirements_accelerator_base_image
    parent: launch-faq
---

アクセラレータを利用するジョブの場合、必要なアクセラレータコンポーネントを含むベースイメージを提供してください。アクセラレータイメージの要件は以下の通りです:

- Debian との互換性 (Launch Dockerfile は Python のインストールに apt-get を使用します)
- サポートされている CPU と GPU のハードウェア命令セット (意図した GPU との CUDA バージョン互換性を確認する)
- 提供されたアクセラレータのバージョンと機械学習アルゴリズムのパッケージ間の互換性
- ハードウェア互換性のために追加の手順が必要なパッケージのインストール