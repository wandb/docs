---
title: W&B サーバーの自動更新を無効にする
description: W&B サーバー の自動アップデートを無効にする方法をご紹介します。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-disable-automatic-app-version-updates
    parent: self-managed
weight: 99
---

このページでは、W&B サーバーの自動バージョンアップグレードを無効にし、バージョンを固定する方法を説明します。ここでの手順は、[W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}}) で管理されているデプロイメントのみに対応しています。

{{% alert %}}
W&B は、主要な W&B サーバーリリースをリリース日から 12 ヶ月間サポートします。**セルフマネージド**インスタンスを利用するお客様は、サポートを維持するため適切なタイミングでアップグレードを行う責任があります。サポートされていないバージョンに留まり続けることは避けてください。W&B では **セルフマネージド**インスタンスをご利用のお客様に対して、サポート維持および最新機能・パフォーマンス改善・修正を受け取るため、少なくとも四半期ごとにデプロイメントを最新リリースにアップデートすることを強く推奨しています。
{{% /alert %}}

## 要件

- W&B Kubernetes Operator `v1.13.0` 以降
- System Console `v2.12.2` 以降

これらの要件を満たしているかどうかは、ご利用中の W&B Custom Resource または Helm chart をご確認ください。`operator-wandb` および `system-console` コンポーネントの `version` 値をご確認ください。

## 自動アップデートの無効化
1. `admin` ロールを持つユーザーとして W&B App にログインします。
2. 画面上部のユーザーアイコンをクリックし、**System Console** を選択します。
3. **Settings** > **Advanced** に進み、**Other** タブを選択します。
4. **Disable Auto Upgrades** セクションで **Pin specific version** をオンにします。
5. **Select a version** のドロップダウンから、W&B サーバーのバージョンを選択します。
6. **Save** をクリックします。

    {{< img src="/images/hosting/disable_automatic_updates_saved_and_enabled.png" alt="Disable Automatic Updates Saved" >}}

    これで自動アップグレードがオフとなり、W&B サーバーは選択したバージョンに固定されます。
1. 自動アップグレードが無効化されていることを確認します。**Operator** タブに移動し、リコンシリエーションログで `Version pinning is enabled` という文字列を検索してください。

```
│info 2025-04-17T17:24:16Z wandb default No changes found
│info 2025-04-17T17:24:16Z wandb default Active spec found
│info 2025-04-17T17:24:16Z wandb default Desired spec
│info 2025-04-17T17:24:16Z wandb default License
│info 2025-04-17T17:24:16Z wandb default Version Pinning is enabled
│info 2025-04-17T17:24:16Z wandb default Found Weights & Biases instance, processing the spec...
│info 2025-04-17T17:24:16Z wandb default === Reconciling Weights & Biases instance...
```