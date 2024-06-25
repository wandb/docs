---
description: あなたのユーザー設定でプロフィール情報、アカウントデフォルト、アラート、ベータ製品への参加、GitHub インテグレーション、ストレージ使用量、アカウントの有効化、およびチームの作成を管理します。
displayed_sidebar: default
---


# User settings

ユーザープロファイルページに移動し、右上のユーザーアイコンを選択します。ドロップダウンから**Settings**を選択します。

### Profile

**Profile**セクションでは、アカウント名や所属機関を管理・変更できます。オプションで経歴、所在地、個人または所属機関のウェブサイトへのリンク、プロファイル画像を追加することができます。

### Project defaults

**Project** **Defaults**セクションでは、アカウントのデフォルトの振る舞いを変更できます。以下の設定を管理できます：

* **Default location to create new projects** - ドロップダウンメニューから新しいデフォルトとして設定するエンティティを選択します。自分のアカウントまたは所属しているチームを指定できます。
* **Default projects privacy in your personal account** - プロジェクトを作成する際に自動的にプロジェクトを公開（誰でも閲覧可能）、非公開（自分だけが閲覧・貢献可能）、またはオープン（誰でもrunsを送信したりReportsを作成可能）に設定します。オプションでチームを作成し、非公開プロジェクトでコラボレーションできます。
* **Enable code savings in your personal account** - Weights and Biasesにデフォルトで最新のgit commitハッシュを保存する許可を与えます。コードの保存を有効にするために、個人アカウントでコード保存を有効にするオプションをトグルします。コードの保存と比較についての詳細は、[Code Saving](../features/panels/code.md)を参照してください。

### Teams

**Team**セクションで新しいチームを作成します。新しいチームを作成するには、**New team**ボタンを選択し、以下の情報を提供します：

* **Team name** - チーム名。チーム名は一意である必要があります。チーム名は変更できません。
* **Team type** - **Work**または**Academic**ボタンを選択します。
* **Company/Organization** - チームの会社または組織の名前を提供します。ドロップダウンメニューから会社または組織を選択します。新しい組織を提供することもできます。

:::info
管理者アカウントのみがチームを作成できます。
:::

### Beta features

**Beta Features**セクションでは、開発中の新製品の楽しいアドオンやスニークプレビューをオプションで有効にできます。有効にしたいベータ機能の隣にあるトグルスイッチを選択します。

### Alerts

runsがクラッシュ、完了、またはカスタムアラートを設定したときに通知を受け取ることができます。[wandb.alert()](../../runs/alert.md)を使って通知を設定します。EメールやSlackを通して通知を受け取ります。通知を受け取りたいイベントタイプの隣にあるスイッチをトグルします。

* **Runs finished**: Weights and Biasesのrunが正常に終了したかどうか。
* **Run crashed**: runが失敗して終了した場合の通知。

通知の設定と管理方法の詳細は、[Send alerts with wandb.alert](../../runs/alert.md)を参照してください。

### Personal GitHub integration

個人のGitHubアカウントを接続します。GitHubアカウントを接続するには：

1. **Connect Github**ボタンを選択します。これにより、オープン認証 (OAuth) ページにリダイレクトされます。
2. **Organization access**セクションでアクセスを許可する組織を選択します。
3. **Authorize** **wandb**を選択します。

### Delete your account

アカウントを削除するには、**Delete Account**ボタンを選択します。

:::caution
アカウント削除は元に戻せません。
:::

### Storage

**Storage**セクションでは、Weights and Biasesサーバーでアカウントが消費した総メモリ使用量を説明します。デフォルトのストレージプランは100GBです。ストレージと価格設定の詳細は、[Pricing](https://wandb.ai/site/pricing)ページを参照してください。