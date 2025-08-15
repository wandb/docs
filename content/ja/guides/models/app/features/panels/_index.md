---
title: パネル
cascade:
- url: guides/app/features/panels/:filename
menu:
  default:
    identifier: ja-guides-models-app-features-panels-_index
    parent: w-b-app-ui-reference
url: guides/app/features/panels
weight: 1
---

workspace パネルの可視化を使って、[記録したデータ]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}})をキーごとに探索し、ハイパーパラメーターと出力メトリクスの関係を可視化するなど、さまざまな用途でご利用いただけます。

## Workspace モード

W&B のプロジェクトは 2 つの workspace モードに対応しています。workspace 名の横にあるアイコンでモードを確認できます。

| アイコン | Workspace モード |
| --- | --- |
| {{< img src="/images/app_ui/automated_workspace.svg" alt="automated workspace icon" width="32px" >}} | **Automated workspaces** はプロジェクト内で記録されたすべてのキーに対して自動でパネルを生成します。自動 workspace を選ぶタイミング:<ul><li>プロジェクトの全データを素早く可視化してスタートしたい時</li><li>記録するキーが少ない小規模なプロジェクトの場合</li><li>広くデータを分析する場合</li></ul>自動 workspace でパネルを削除した場合でも [Quick add]({{< relref path="#quick-add" lang="ja" >}}) 機能で再作成できます。 |
| {{<img src="/images/app_ui/manual_workspace.svg" alt="manual workspace icon" width="32px" >}} | **Manual workspaces** は最初は空の状態で、ユーザーが意図的に追加したパネルのみ表示されます。手動 workspace を選ぶタイミング:<ul><li>プロジェクトで記録したキーの一部だけを重視したい場合</li><li>特定の分析に集中したい場合</li><li>不要なパネルの読み込みを避けて workspace のパフォーマンスを上げたい場合</li></ul>[Quick add]({{< relref path="#quick-add" lang="ja" >}}) で便利な可視化をセクションごとにすばやく追加できます。 |

workspace のパネル生成方法を変更するには、[workspace をリセット]({{< relref path="#reset-a-workspace" lang="ja" >}})してください。

{{% alert title="workspace の変更を元に戻す" %}}
workspace の変更を元に戻すには、Undo ボタン（左向き矢印）をクリックするか **CMD + Z**（macOS）または **CTRL + Z**（Windows / Linux）を入力します。
{{% /alert %}}

## workspace をリセットする

workspace をリセットするには:

1. workspace 上部のアクションメニュー `...` をクリックします。
1. **Reset workspace** を選択します。

## workspace レイアウトの設定 {#configure-workspace-layout}

workspace のレイアウトを設定するには、workspace 上部の **Settings** をクリックし、**Workspace layout** を選択します。

- **Hide empty sections during search**（標準で有効）
- **Sort panels alphabetically**（標準では無効）
- **Section organization**（標準では最初のプレフィックスでグループ化）。この設定を変更するには:
  1. 鍵アイコンをクリックします。
  1. セクション内でパネルをグループ化する方法を選択します。

workspace 内の折れ線グラフのデフォルト設定は [Line plots]({{< relref path="line-plot/#all-line-plots-in-a-workspace" lang="ja" >}}) をご参照ください。

### セクションのレイアウト設定 {#configure-section-layout}

セクションのレイアウトを設定するには、ギアアイコンをクリックし、**Display preferences** を選択します。
- **ツールチップに色付き run 名を表示するかどうか**（標準で有効）
- **コンパニオンチャートのツールチップにはハイライトされた run のみ表示**（標準では無効）
- **ツールチップに表示する run 数**（単一 run・全 run・**Default** から選択）
- **メインチャートのツールチップに完全な run 名を表示する**（標準では無効）

## パネルをフルスクリーンで表示

フルスクリーンモードでは、run セレクターが表示され、パネルは通常 1,000 バケットではなく 10,000 バケットの高解像度サンプリングプロットになります。

パネルをフルスクリーンで見る手順:

1. パネルにカーソルを合わせます。
1. パネルのアクションメニュー `...` をクリックし、フルスクリーンボタン（ビューファインダー、または四隅を示すアウトライン）を選択します。
    {{< img src="/images/app_ui/panel_fullscreen.png" alt="Full-screen panel" >}}
1. [パネルを共有]({{< relref path="#share-a-panel" lang="ja" >}})する際にフルスクリーン表示していれば、リンクを開いた時も自動的にフルスクリーンで表示されます。

フルスクリーンモードから workspace へ戻るには、ページ上部の左向き矢印をクリックしてください。

## パネルを追加する

このセクションでは、workspace にパネルを追加するさまざまな方法を説明します。

### パネルを手動で追加

パネルは全体またはセクション単位で 1 つずつ追加できます。

1. 全体にパネルを追加する場合、パネル検索フィールドの近くにあるコントロールバーで **Add panels** をクリックします。
1. 特定のセクションに直接追加する場合、そのセクションのアクションメニュー `...` をクリックし、**+ Add panels** を選択します。
1. チャートなど追加したいパネルタイプを選びます。パネルの設定画面がデフォルト値で表示されます。
1. 任意でパネルや表示設定をカスタマイズします。設定項目はパネルタイプにより異なります。各パネルタイプの詳細オプションは、[Line plots]({{< relref path="line-plot/" lang="ja" >}}) や [Bar plots]({{< relref path="bar-plot.md" lang="ja" >}}) など関連セクションをご参照ください。
1. **Apply** をクリックします。

{{< img src="/images/app_ui/add_single_panel.gif" alt="パネル追加デモ" >}}

### Quick add パネル {#quick-add}

**Quick add** を使うと、選んだキーごとに自動でパネルを追加できます。全体にもセクション単位にも適用できます。

{{% alert %}}
自動 workspace でパネルが一つも削除されていない場合、**Quick add** オプションは表示されません。なぜなら既にすべての記録キーに対応するパネルが揃っているからです。削除したパネルを再追加する場合のみご利用いただけます。
{{% /alert %}}

1. 全体に **Quick add** を使うには、パネル検索フィールド近くの **Add panels** をクリックし、続けて **Quick add** を選びます。
1. セクション単位で **Quick add** を使うには、そのセクションのアクションメニュー `...` をクリック、**Add panels** を押し、**Quick add** を選びます。
1. パネル一覧が表示されます。チェックマーク付きはすでに workspace にあるパネルです。
    - 全パネルを一括追加する場合は、リスト上部の **Add <N> panels** ボタンをクリックしてください。**Quick Add** リストが閉じ、workspace に新パネルが反映されます。
    - 一つずつ追加したい場合は、パネル行にカーソルを合わせて **Add** をクリック。必要なだけ繰り返し、最後に右上の **X** を押して **Quick Add** リストを閉じます。新パネルが workspace に表示されます。
1. 必要に応じてパネルの設定をカスタマイズします。

## パネルを共有する

このセクションではリンクを使ってパネルを共有する方法を説明します。

リンクでパネルを共有するには、以下いずれかの手順をとります:

- パネルをフルスクリーン表示中に、ブラウザの URL をコピー
- アクションメニュー `...` をクリックし、**Copy panel URL** を選択

ユーザーまたはチームにリンクを共有してください。リンクにアクセスすると、パネルは [フルスクリーン表示]({{< relref path="#view-a-panel-in-full-screen-mode" lang="ja" >}})で開きます。

フルスクリーン表示から workspace へ戻るには、ページ上部の左向き矢印をクリックします。

### プログラムでパネルのフルスクリーンリンクを生成する

たとえば [オートメーションの作成]({{< relref path="/guides/core/automations/" lang="ja" >}}) 時など、パネルのフルスクリーン URL を自動で扱いたい場合があります。このときの URL 形式は下記です。例の `<ENTITY_NAME>`、`<PROJECT_NAME>`、`<PANEL_NAME>`、`<SECTON_NAME>` は適宜置き換えてください。

```text
https://wandb.ai/<ENTITY_NAME>/<PROJECT_NAME>?panelDisplayName=<PANEL_NAME>&panelSectionName=<SECTON_NAME>
```

同じセクション内で同名パネルが複数ある場合、この URL は最初のパネルに遷移します。

### パネルを埋め込む・SNS で共有する

パネルをウェブサイトに埋め込んだり SNS で共有したい場合、リンクを知っていれば誰でも閲覧できる設定が必要です。プロジェクトが非公開の場合、参加メンバーのみが閲覧可能です。プロジェクトが公開設定の場合、リンクを知っていれば誰でも閲覧できます。

パネルを埋め込んだり SNS で共有するためのコードを取得するには:

1. workspace でパネルにカーソルを合わせ、アクションメニュー `...` をクリック
1. **Share** タブを選択
1. **Only those who are invited have access** を **Anyone with the link can view** に変更（この設定にしないと次の手順の選択肢が出ません）
1. **Share on Twitter**、**Share on Reddit**、**Share on LinkedIn**、**Copy embed link** の中から選択

### パネルをレポートとしてメール送信する

1 パネルを単独のレポートとしてメール送信するには:
1. パネルにカーソルを合わせ、アクションメニュー `...` をクリック
1. **Share panel in report** を選択
1. **Invite** タブを選択
1. メールアドレスまたはユーザー名を入力
1. 必要であれば **can view** を **can edit** に変更
1. **Invite** をクリック。共有先ユーザーに、パネルだけが含まれたレポートへのリンク付きメールが送信されます。

[パネルを共有]({{< relref path="#share-a-panel" lang="ja" >}}) した時と異なり、このレポートから workspace には遷移できません。

## パネルの管理

### パネルを編集

パネルを編集するには:

1. 鉛筆アイコンをクリック
1. パネル設定を変更
1. パネルタイプを変更する場合はタイプを選択し、設定を行う
1. **Apply** をクリック

### パネルを移動

パネルを他セクションへ移動する場合、パネルのドラッグハンドルを利用できます。リストから新しいセクションを選びたい場合は:

1. 必要に応じて **Add section** をクリックして新セクションを作成
1. パネルのアクションメニュー `...` をクリック
1. **Move** を選択し、新しいセクションを選ぶ

同一セクション内ではドラッグハンドルを使ってパネルの並び順も変更できます。

### パネルを複製

パネルを複製するには:

1. パネル上部のアクションメニュー `...` をクリック
1. **Duplicate** を選択

必要であれば複製後に [カスタマイズ]({{< relref path="#edit-a-panel" lang="ja" >}}) や [移動]({{< relref path="#move-a-panel" lang="ja" >}}) もできます。

### パネルを削除

パネルを削除するには:

1. パネルにカーソルを合わせます
1. アクションメニュー `...` を選択
1. **Delete** をクリック

manual workspace からすべてのパネルを削除したい場合は、その workspace のアクションメニュー `...` から **Clear all panels** を選択してください。

自動・手動いずれの workspace でも、[workspace をリセット]({{< relref path="#reset-a-workspace" lang="ja" >}}) することで全てのパネルを削除できます。**Automatic** を選べばデフォルトパネルがある状態から、**Manual** を選べば空の workspace から開始できます。

## セクションの管理

デフォルトでは、workspace のセクションはキーの記録階層に基づきます。ただし manual workspace の場合はパネルを追加し始めてからセクションが現れます。

### セクションを追加

セクションを追加したい時は、リストの最後にある **Add section** をクリックしてください。

既存セクションの前後に追加したい場合は、該当セクションのアクションメニュー `...` から **New section below** または **New section above** を選んでください。

### セクション内パネルの管理

多数のパネルがあるセクションは自動でページネーションされます。ページあたりのデフォルト表示パネル数はパネルの設定やサイズで決まります。

{{% alert %}}
**Custom grid** レイアウトはまもなく廃止されます。W&B では今後 Custom grid の利用を推奨しません。なるべく **Standard grid** へアップデートしてください。

**Custom grid** 廃止後は workspace のレイアウトはすべて **Standard grid** へ統一され、カスタマイズ不可となります。
{{% /alert %}}

1. セクションがどのレイアウトを使っているか確認するには、セクションのアクションメニュー `...` をクリックします。レイアウトを変更したい場合は **Layout grid** セクション内で **Standard grid** または **Custom grid** を選択してください。
1. パネルのサイズ変更は、カーソルを合わせドラッグハンドルを掴んでサイズを調整
  - **Standard grid** 利用時は 1 パネルでもリサイズするとセクション内すべてのパネルサイズが変わります。
  - **Custom grid** 利用時は各パネルごとにサイズを調整可能
1. ページネーションされている場合、1 ページで表示するパネル数は以下の手順で調整可能:
  1. セクション上部の **1 to <X> of <Y>** をクリック（`<X>` は表示パネル数、 `<Y>` は総パネル数）
  1. 1 ページ当たりの表示数を最大 100 まで選択
1. セクションからパネルを削除したい場合:
  1. パネルにカーソルを合わせ、アクションメニュー `...` をクリック
  1. **Delete** を選択

workspace を自動モードでリセットすると削除したパネルもすべて再表示されます。

### セクション名の変更

セクション名を変更するには、アクションメニュー `...` から **Rename section** を選択します。

### セクションの削除

セクションを削除するには、`...` メニューから **Delete section** を選択してください。これにより、該当セクションとその中のパネルが全て削除されます。