---
slug: /guides/hosting
displayed_sidebar: ja
---

# プライベートホスティング

## W&Bホスティングオプション​

:::info
W&Bサーバーを貴社のインフラストラクチャでプライベートホスティングする前に、wandb.aiクラウドを使用することをお勧めします。クラウドはシンプルかつ安全で、設定は必要ありません。詳細については[こちら](https://docs.wandb.ai/quickstart) をクリックしてください。
:::

W&Bサーバーをプロダクション環境でセットアップする方法は、主として3種類あります：

1. [プロダクションクラウド](setup/private-cloud.md): プライベートクラウドで、W&Bが提供するterraformスクリプトを使って、わずか数ステップでプロダクション展開をセットアップできます。
2. [専用クラウド](setup/dedicated-cloud.md): お好きなクラウドリージョンで、W&Bのシングルテナントインフラストラクチャーでの専用のマネージド展開。
3. [オンプレミス / ベアメタル](setup/on-premise-baremetal.md): W&Bは、企業オンプレミスデータセンター内の多くのベアメタルサーバー上でのプロダクションサーバーのセットアップをサポートしています。wandbサーバーを実行して、ローカルインフラストラクチャーでW&Bのホスティングを簡単に開始することで、迅速に作業を開始できます。

## W&Bサーバークイックスタート​

1.  [Docker](https://www.docker.com)と[Python](https://www.python.org) がインストールされたマシン上で以下を実行します：

    ```
    pip install wandb
    wandb server start
    ```
2. [Deployer](https://deploy.wandb.ai/)から無料ライセンスを生成します。
3. これをローカル設定に追加します。

localhostの` /system-admin`ページでライセンスを貼り付けます

![Copy your license from Deployer and paste it into your Local settings](@site/static/images/hosting/License.gif)
