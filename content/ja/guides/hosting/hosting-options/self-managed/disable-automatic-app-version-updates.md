---
title: W&B サーバーの自動アップデートを無効にする
description: W&B サーバーの自動アップデートを無効にする方法について説明します。
menu:
  default:
    identifier: disable-automatic-app-version-updates
    parent: self-managed
weight: 99
---

このページでは、W&B Server の自動バージョンアップグレードを無効にし、バージョンをピン留めする方法について説明します。これらの手順は [W&B Kubernetes Operator]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/" >}}) によって管理されているデプロイメントのみで有効です。

{{% alert %}}
W&B は、主要な W&B Server リリースのリリース日から 12 ヶ月間サポートします。**Self-managed** インスタンスをご利用のお客様は、サポートを継続するために適宜アップグレードする責任があります。サポート対象外のバージョンを使い続けることは避けてください。**Self-managed** インスタンスをご利用のお客様には、サポート維持と最新の機能・パフォーマンス改善・修正を受けるために、四半期ごとに最低一度は最新リリースへアップデートすることを強く推奨します。
{{% /alert %}}

## 必要条件

- W&B Kubernetes Operator `v1.13.0` 以上
- System Console `v2.12.2` 以上

これらの要件を満たしているか確認するには、ご利用中のインスタンスの W&B Custom Resource もしくは Helm チャートを参照してください。`operator-wandb` および `system-console` コンポーネントの `version` 値を確認します。

## 自動アップデートの無効化
1. `admin` ロールを持つユーザーとして W&B App にログインします。
2. 上部のユーザーアイコンをクリックし、**System Console** をクリックします。
3. **Settings** > **Advanced** に移動し、**Other** タブを選択します。
4. **Disable Auto Upgrades** セクションで **Pin specific version** をオンにします。
5. **Select a version** ドロップダウンをクリックし、W&B Server バージョンを選びます。
6. **Save** をクリックします。

    {{< img src="/images/hosting/disable_automatic_updates_saved_and_enabled.png" alt="Disable Automatic Updates Saved" >}}

    これで自動アップグレードが無効化され、選択したバージョンで W&B Server をピン留めできます。
1. 自動アップグレードが無効化されていることを確認します。**Operator** タブに移動し、和解ログから `Version pinning is enabled` という文字列を検索します。

```
│情報 2025-04-17T17:24:16Z wandb default 変更は見つかりませんでした
│情報 2025-04-17T17:24:16Z wandb default アクティブな spec が見つかりました
│情報 2025-04-17T17:24:16Z wandb default 希望の spec
│情報 2025-04-17T17:24:16Z wandb default ライセンス
│情報 2025-04-17T17:24:16Z wandb default Version Pinning is enabled
│情報 2025-04-17T17:24:16Z wandb default Weights & Biases インスタンスが見つかりました。spec をプロセッシング中...
│情報 2025-04-17T17:24:16Z wandb default === Weights & Biases インスタンスを調整中...
```