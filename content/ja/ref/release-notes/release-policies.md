---
title: リリース方針とプロセス
description: W&B サーバーのリリースプロセス
date: 2025-05-01
menu:
  default:
    identifier: server-release-process
    parent: w-b-platform
  reference:
    identifier: ja-ref-release-notes-release-policies
weight: 20
---

このページでは、W&B Server のリリース内容および W&B のリリースポリシーについて詳しく説明しています。このページは [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) および [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) デプロイメントに関するものです。各 W&B Server リリースの詳細については、[W&B リリースノート]({{< relref path="/ref/release-notes/" lang="ja" >}}) をご参照ください。

W&B は [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を完全に管理しており、このページの内容は該当しません。

## リリースサポートとサポート終了（EOL）ポリシー
W&B は、各メジャー W&B Server リリースについてリリース日から 12 ヶ月間サポートします。
- **Dedicated Cloud** インスタンスはサポートを維持するため、自動的に更新されます。
- **Self-managed** インスタンスをご利用の場合は、お客様ご自身で適切なタイミングでアップグレードし、サポート対象を維持してください。サポート対象外のバージョンを使い続けるのは避けてください。

  {{% alert %}}
  **Self-managed** インスタンスをご利用のお客様には、最低でも四半期に一度は最新リリースにアップデートし、サポートの維持・最新機能・パフォーマンス改善・修正を受けることを強く推奨します。
  {{% /alert %}}

## リリースの種類と頻度
- **メジャーリリース** は月次で提供されます。新機能、機能強化、パフォーマンス改善、中・低レベルのバグ修正、非推奨事項が含まれます。例: `0.68.0`
- **パッチリリース** はメジャーバージョン内で必要に応じて提供され、重大または高レベルのバグ修正を含みます。例: `0.67.1`

## リリースのロールアウト
1. テストおよび検証が完了後、まず **Dedicated Cloud** インスタンスすべてにリリースが適用され、最新の状態が維持されます。
1. 追加の観察期間を経て、リリースが公開され、**Self-managed** デプロイメントはお客様のスケジュールでアップグレードできます。アップグレードのタイミングは [リリースサポートとサポート終了（EOL）ポリシー]({{< relref path="#release-support-and-end-of-life-policy" lang="ja" >}}) に従ってください。[W&B Server のアップグレード方法]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}}) について詳しくはこちら。

## アップグレード時のダウンタイム
- **Dedicated Cloud** インスタンスのアップグレード時、通常ダウンタイムは想定されていませんが、以下の場合に発生する可能性があります。
  - 新機能や機能拡張により、コンピュート・ストレージ・ネットワークなど基盤インフラに変更が必要となる場合
  - 重大なインフラ変更（セキュリティ修正 等）を展開する場合
  - 現在のバージョンが[サポート終了（EOL）]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}}) となっており、サポート維持のために W&B によってアップグレードされる場合
- **Self-managed** デプロイメントの場合、お客様側でサービスレベル目標（SLO）を満たすローリングアップデートプロセスをご用意ください。例: [Kubernetes 上で W&B Server を実行する方法]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}})

## 機能の利用可能タイミング
インストールまたはアップグレード直後、一部の機能はすぐに利用できない場合があります。

### エンタープライズ機能
エンタープライズライセンスには、重要なセキュリティ機能や企業向け機能のサポートが含まれます。一部の高度な機能はエンタープライズライセンスが必要です。

- **Dedicated Cloud** はエンタープライズライセンスが含まれており、追加の操作は必要ありません。
- **Self-managed** デプロイメントでは、エンタープライズライセンスが設定されるまでは該当機能は利用できません。詳細は [W&B Server ライセンスの取得方法]({{< relref path="/guides/hosting/hosting-options/self-managed.md#obtain-your-wb-server-license" lang="ja" >}}) をご覧ください。

### プライベートプレビューおよびオプトイン機能
ほとんどの機能は W&B Server のインストールまたはアップグレード直後に利用できます。一部の機能については、W&B チームによる有効化が必要です。

{{% alert color="warning" %}}
プレビュー段階の機能は今後変更される可能性があります。プレビュー機能が必ず一般公開されるとは限りません。
{{% /alert %}}

- **プライベートプレビュー**: W&B はデザインパートナーやアーリーアダプターへこれら機能を案内し、フィードバックを募集します。プライベートプレビュー機能はプロダクション環境では推奨されません。

    プライベートプレビュー機能の利用には、W&B チームによる機能有効化が必要です。一般向けドキュメントはなく、個別に案内されます。インターフェースや API が変更される場合や、機能が完全には実装されていないことがあります。
- **パブリックプレビュー**: 一般公開前にパブリックプレビューを試すには、W&B にご連絡のうえオプトイン登録を行ってください。

    パブリックプレビュー機能も、W&B チームによる有効化が必要です。ドキュメントが不完全な場合があります。また、インターフェースや API が変更される場合や、機能が完全には実装されていないことがあります。

各 W&B Server リリースの詳細・制限事項などについては、[W&B リリースノート]({{< relref path="/ref/release-notes/" lang="ja" >}}) をご参照ください。