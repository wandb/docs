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

workspace の パネル 可視化 を使って、[ログ した データ]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}) を キー ごとに 探索 し、ハイパーパラメーター と 出力 メトリクス の 関係 を 可視化 する など が できます。

## workspace modes

W&B の Projects は 2 種類 の workspace モード をサポートします。workspace 名 の 横 の アイコン が その モード を 示します。

| アイコン | workspace モード |
| --- | --- |
| {{< img src="/images/app_ui/automated_workspace.svg" alt="自動 workspace アイコン" width="32px" >}} | **Automated workspaces** は、Project 内 で ログ された すべて の キー に対する パネル を 自動生成 します。自動 workspace を 選ぶ と よい 場合:<ul><li>Project の 利用可能 な すべて の データ を すばやく 可視化 して 始めたい とき。</li><li>ログ する キー が 少ない 小規模 な Project の とき。</li><li>より 広範 な 分析 を したい とき。</li></ul>自動 workspace で パネル を 削除 した 場合 は、[Quick add]({{< relref path="#quick-add" lang="ja" >}}) で 再作成 できます。 |
| {{<img src="/images/app_ui/manual_workspace.svg" alt="手動 workspace アイコン" width="32px" >}} | **Manual workspaces** は まっさら な 状態 から 始まり、User が 意図的 に 追加 した パネル だけ を 表示 します。手動 workspace を 選ぶ と よい 場合:<ul><li>Project で ログ された キー の 一部 に だけ 関心 が ある とき。</li><li>より 焦点 を 絞った 分析 を したい とき。</li><li>自分 に とって 有用性 の 低い パネル の 読み込み を 避け、workspace の パフォーマンス を 改善 したい とき。</li></ul>[Quick add]({{< relref path="#quick-add" lang="ja" >}}) を 使う と、手動 workspace と その セクション に 有用 な 可視化 を すばやく 追加 できます。 |

workspace が パネル を 生成 する 方法 を 変更 する に は、[workspace を リセット する]({{< relref path="#reset-a-workspace" lang="ja" >}}) を 参照してください。

{{% alert title="workspace への 変更 を 元に戻す" %}}
workspace への 変更 を 元に戻す に は、Undo ボタン（左向き の 矢印）を クリック する か、**CMD + Z**（macOS）または **CTRL + Z**（Windows / Linux）を 入力 します。
{{% /alert %}}

## workspace を リセット する

workspace を リセット する 手順:

1. workspace の 上部 で アクション メニュー `...` を クリック します。
1. **Reset workspace** を クリック します。

## workspace レイアウト を 設定 する {#configure-workspace-layout}

workspace の レイアウト を 設定 する に は、workspace の 上部 付近 の **Settings** を クリック し、**Workspace layout** を クリック します。

- **Hide empty sections during search**（デフォルト で オン）
- **Sort panels alphabetically**（デフォルト で オフ）
- **Section organization**（デフォルト では 最初 の プレフィックス で グループ化）。この 設定 を 変更 する に は:
  1. 錠前 アイコン を クリック します。
  1. セクション 内 で パネル を どの よう に グループ化 する か を 選択 します。

workspace 内 の 折れ線 グラフ の 既定 設定 に ついて は、[Line plots]({{< relref path="line-plot/#all-line-plots-in-a-workspace" lang="ja" >}}) を 参照 してください。

### セクション の レイアウト を 設定 する {#configure-section-layout}

セクション の レイアウト を 設定 する に は、歯車 アイコン を クリック し、**Display preferences** を クリック します。
- **Turn on or off colored run names in tooltips**（デフォルト で オン）
- **Only show highlighted run in companion chart tooltips**（デフォルト で オフ）
- **Number of runs shown in tooltips**（単一 の run、すべて の run、または **Default**）
- **Display full run names on the primary chart tooltip**（デフォルト で オフ）

## パネル を 全画面 モード で 表示 する

全画面 モード では、run セレクター が 表示 され、パネル は 10,000 バケット の フル フィデリティ サンプリング モード の プロット を 使用 します（通常 は 1000 バケット）。

パネル を 全画面 モード で 表示 する 手順:

1. パネル に マウス オーバー します。
1. パネル の アクション メニュー `...` を クリック し、ビューファインダー（正方形 の 四隅 の アウトライン）に 見える 全画面 ボタン を クリック します。
    {{< img src="/images/app_ui/panel_fullscreen.png" alt="全画面 パネル" >}}
1. 全画面 モード で 表示 中 に [パネル を 共有]({{< relref path="#share-a-panel" lang="ja" >}}) すると、生成 される リンク は 自動的 に 全画面 モード で 開きます。

全画面 モード から パネル の workspace に 戻る に は、ページ 上部 の 左向き 矢印 を クリック します。全画面 モード を 退出 せず に セクション 内 の パネル を 移動 する に は、パネル 下 の **Previous** と **Next** ボタン、または キーボード の 左右 矢印 キー を 使用 します。

## パネル を 追加 する

この セクション では、workspace に パネル を 追加 する さまざま な 方法 を 説明 します。

### パネル を 手動 で 追加 する

パネル は グローバル に も セクション 単位 でも、1 つ ずつ workspace に 追加 できます。

1. グローバル に パネル を 追加 する に は、パネル 検索 フィールド 付近 の コントロール バー に ある **Add panels** を クリック します。
1. セクション に 直接 追加 する に は、その セクション の アクション メニュー `...` を クリック し、**+ Add panels** を クリック します。
1. 追加 する パネル の 種類（例: チャート）を 選択 します。パネル の 設定 詳細 が デフォルト 値 と ともに 表示 されます。
1. 必要 に 応じて、パネル と 表示 設定 を カスタマイズ します。利用 可能 な 設定 は 選択 した パネル の 種類 に よって 異なります。各 パネル 種類 の オプション の 詳細 は、[Line plots]({{< relref path="line-plot/" lang="ja" >}}) や [Bar plots]({{< relref path="bar-plot.md" lang="ja" >}}) など の 該当 セクション を 参照 してください。
1. **Apply** を クリック します。

{{< img src="/images/app_ui/add_single_panel.gif" alt="パネル 追加 の デモ" >}}

### Quick add {#quick-add}

**Quick add** を 使う と、選択 した 各 キー に 対する パネル を 自動的 に 追加 できます（グローバル または セクション 単位）。

{{% alert %}}
削除 済み の パネル が ない 自動 workspace では、すでに すべて の ログ 済み キー の パネル が 含まれて いる ため **Quick add** オプション は 表示 されません。削除 した パネル を 再追加 する 場合 に **Quick add** を 使用 できます。
{{% /alert %}}

1. **Quick add** で パネル を グローバル に 追加 する に は、パネル 検索 フィールド 付近 の コントロール バー で **Add panels** を クリック し、**Quick add** を クリック します。
1. **Quick add** で パネル を セクション に 追加 する に は、その セクション の アクション メニュー `...` を クリック し、**Add panels** を クリック して から **Quick add** を クリック します。
1. パネル の 一覧 が 表示 されます。チェックマーク の 付いた パネル は、すでに workspace に 含まれて います。
    - すべて の 利用 可能 な パネル を 追加 する に は、一覧 上部 の **Add <N> panels** ボタン を クリック します。**Quick Add** リスト が 閉じられ、新しい パネル が workspace に 表示 されます。
    - 個別 の パネル を 追加 する に は、その 行 に マウス オーバー して **Add** を クリック します。追加 したい パネル ごと に この 手順 を 繰り返し、右上 の **X** を クリック して **Quick Add** リスト を 閉じます。新しい パネル が workspace に 表示 されます。
1. 必要 に 応じて、パネル の 設定 を カスタマイズ します。

## パネル を 共有 する

この セクション では、リンク を 使って パネル を 共有 する 方法 を 説明 します。

リンク で パネル を 共有 する 方法 は 次 の いずれか です。

- パネル を 全画面 モード で 表示 中 に、ブラウザー の URL を コピー します。
- アクション メニュー `...` を クリック し、**Copy panel URL** を 選択 します。

リンク を User または Team と 共有 します。リンク に アクセス すると、パネル は [全画面 モード]({{< relref path="#view-a-panel-in-full-screen-mode" lang="ja" >}}) で 開きます。

全画面 モード から パネル の workspace に 戻る に は、ページ 上部 の 左向き 矢印 を クリック します。

### パネル の 全画面 リンク を プログラム で 作成 する
[オートメーション を 作成]({{< relref path="/guides/core/automations/" lang="ja" >}}) する とき など、パネル の 全画面 URL を 含める と 便利 な 場合 が あります。ここ では パネル の 全画面 URL の 形式 を 示します。以下 の 例 では、角括弧 内 の entity、project、panel、section 名 を 置き換えて ください。

```text
https://wandb.ai/<ENTITY_NAME>/<PROJECT_NAME>?panelDisplayName=<PANEL_NAME>&panelSectionName=<SECTON_NAME>
```

同じ セクション 内 に 同名 の パネル が 複数 ある 場合、この URL は 最初 の 同名 パネル を 開きます。

### パネル を 埋め込む / SNS で 共有 する
Web サイト に パネル を 埋め込む、または SNS で 共有 する に は、リンク を 知って いる すべて の 人 が パネル を 表示 できる 必要 が あります。Project が プライベート の 場合、その Project の メンバー のみ が パネル を 表示 できます。Project が パブリック の 場合、リンク を 知って いる すべて の 人 が パネル を 表示 できます。

SNS で 共有 したり 埋め込み 用 の コード を 取得 する 手順:

1. workspace で パネル に マウス オーバー し、その アクション メニュー `...` を クリック します。
1. **Share** タブ を クリック します。
1. **Only those who are invited have access** を **Anyone with the link can view** に 変更 します。そうしない と、次 の ステップ の 選択肢 は 利用 できません。
1. **Share on Twitter**、**Share on Reddit**、**Share on LinkedIn**、または **Copy embed link** を 選択 します。

### パネル を Report として メール 送信 する
単一 の パネル を スタンドアロン の Report として メール 送信 する に は:
1. パネル に マウス オーバー し、パネル の アクション メニュー `...` を クリック します。
1. **Share panel in report** を クリック します。
1. **Invite** タブ を 選択 します。
1. メール アドレス または ユーザー名 を 入力 します。
1. 必要 に 応じて、**can view** を **can edit** に 変更 します。
1. **Invite** を クリック します。W&B から 共有 先 の User に、共有 した パネル のみ が 含まれる Report への クリック 可能 な リンク が 記載 された メール が 送信 されます。

[パネル を 共有]({{< relref path="#share-a-panel" lang="ja" >}}) した 場合 と は 異なり、この Report から 受信者 が workspace に 移動 する こと は できません。

## パネル を 管理 する

### パネル を 編集 する

パネル の 編集 手順:

1. 鉛筆 アイコン を クリック します。
1. パネル の 設定 を 変更 します。
1. 別 の 種類 の パネル に 変更 する 場合 は、種類 を 選択 して から 設定 を 行います。
1. **Apply** を クリック します。

### パネル を 移動 する

パネル を 別 の セクション に 移動 する に は、パネル の ドラッグ ハンドル を 使用 できます。リスト から 新しい セクション を 選ぶ 場合 は 次 の 手順 を 実行 します。

1. 必要 で あれば、最後 の セクション の 後 に **Add section** を クリック して 新しい セクション を 作成 します。
1. パネル の アクション メニュー `...` を クリック します。
1. **Move** を クリック し、新しい セクション を 選択 します。

ドラッグ ハンドル を 使って、セクション 内 の パネル の 並び順 を 変更 する こと も できます。

### パネル を 複製 する

パネル を 複製 する 手順:

1. パネル 上部 の アクション メニュー `...` を クリック します。
1. **Duplicate** を クリック します。

必要 で あれば、複製 した パネル を [カスタマイズ]({{< relref path="#edit-a-panel" lang="ja" >}}) したり [移動]({{< relref path="#move-a-panel" lang="ja" >}}) したり できます。

### パネル を 削除 する

パネル を 削除 する 手順:

1. パネル に マウス を 乗せます。
1. アクション メニュー `...` を 選択 します。
1. **Delete** を クリック します。

手動 workspace から すべて の パネル を 削除 する に は、その アクション メニュー `...` を クリック し、**Clear all panels** を クリック します。

自動 または 手動 workspace から すべて の パネル を 削除 する 別 の 方法 として、[workspace を リセット]({{< relref path="#reset-a-workspace" lang="ja" >}}) できます。**Automatic** を 選ぶ と 既定 の パネル セット から、**Manual** を 選ぶ と パネル の ない 空 の workspace から 始められます。

## セクション を 管理 する

デフォルト では、workspace の セクション は キー の ログ 階層 を 反映 します。ただし、手動 workspace では、パネル の 追加 を 開始 して 初めて セクション が 表示 されます。

### セクション を 追加 する

セクション を 追加 する に は、最後 の セクション の 後 に **Add section** を クリック します。

既存 セクション の 前後 に 新しい セクション を 追加 する 場合 は、その セクション の アクション メニュー `...` を クリック し、**New section below** または **New section above** を クリック します。

### セクション 内 の パネル を 管理 する
多数 の パネル を 含む セクション は、デフォルト で ページ分割 されます。ページ あたり の 既定 の パネル 数 は、パネル の 設定 や セクション 内 の パネル サイズ に 依存 します。

{{% alert %}}
**Custom grid** レイアウト は まもなく 廃止 予定 です。W&B は **Custom grid** レイアウト の 使用 を 推奨 しません。workspace を **Custom grid** から **Standard grid** に 更新 する こと を 検討 してください。

**Custom grid** レイアウト が 廃止 される と、workspace は **Standard grid** レイアウト に 更新 され、この レイアウト は 今後 変更 不可 に なります。
{{% /alert %}}

1. セクション が どの レイアウト を 使用 して いる か を 確認 する に は、その セクション の アクション メニュー `...` を クリック します。レイアウト を 変更 する に は、**Layout grid** セクション で **Standard grid** または **Custom grid** を 選択 します。
1. パネル の サイズ を 変更 する に は、パネル に マウス オーバー し、ドラッグ ハンドル を クリック して サイズ が 変わる よう に ドラッグ します。
  - セクション が **Standard grid** の 場合、1 つ の パネル を リサイズ すると セクション 内 の すべて の パネル が リサイズ されます。
  - セクション が **Custom grid** の 場合、各 パネル の サイズ を 個別 に カスタマイズ できます。
1. セクション が ページ分割 されて いる 場合、1 ページ に 表示 する パネル 数 を カスタマイズ できます:
  1. セクション 上部 の **1 to <X> of <Y>**（`<X>` は 表示 中 の パネル 数、`<Y>` は 総 パネル 数）を クリック します。
  1. 1 ページ に 表示 する パネル 数 を 最大 100 まで から 選択 します。
1. セクション から パネル を 削除 する に は:
  1. パネル に マウス オーバー し、その アクション メニュー `...` を クリック します。
  1. **Delete** を クリック します。
  
workspace を 自動 workspace に リセット すると、削除 した すべて の パネル が 再び 表示 されます。

### セクション 名 を 変更 する

セクション 名 を 変更 する に は、アクション メニュー `...` を クリック し、**Rename section** を クリック します。

### セクション を 削除 する

セクション の `...` メニュー を 開き、**Delete section** を クリック します。これ に より、その セクション と その 中 の パネル が 削除 されます。