---
title: GitHub Action の変更をテストする
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # エージェント向けプロンプト: wandb/docs における GitHub Actions の変更をテストする
</div>

<div id="requirements">
  ## 要件
</div>

* **W&amp;B 社員としてのアクセス権**: W&amp;B の社内システムにアクセスできる W&amp;B 社員である必要があります。
* **GitHub フォーク**: ワークフローの変更をテストするための wandb/docs の個人用フォーク。フォークでは、デフォルトブランチへのプッシュ権限と、ブランチ保護ルールを回避する権限が必要です。

<div id="agent-prerequisites">
  ## エージェント の前提条件
</div>

開始前に、次の情報を確認してください。

1. **GitHub ユーザー名** - まずフォークのリモートを `git remote -v` で確認し、次にユーザー名を `git config` で確認します。どちらでも見つからない場合にのみ、ユーザーに確認してください。
2. **Fork のステータス** - wandb/docs のフォークを所有しており、デフォルトブランチへの プッシュ 権限とブランチ保護をバイパスする権限があることを確認してください。
3. **テスト範囲** - どの変更をテストしているのかを確認してください (依存関係のアップグレード、機能変更など) 。

<div id="task-overview">
  ## タスクの概要
</div>

`wandb/docs` リポジトリ内の GitHub Actions ワークフローに加えた変更をテストします。

<div id="context-and-constraints">
  ## 前提条件と制約
</div>

<div id="repository-setup">
  ### リポジトリのセットアップ
</div>

* **メインリポジトリ**: `wandb/docs` (origin) 
* **テスト用フォーク**: `<username>/docs` (fork リモート)  - `git remoter -v` だけでははっきりしない場合は、ユーザーにフォークの Endpoint を確認してください。
* **重要**: PR の GitHub Actions は常に PR ブランチではなく、ベースブランチ (main) から実行されます。
* **Mintlify デプロイの制限**: Mintlify のデプロイと `link-rot` チェックは、フォークではなくメインの `wandb/docs` リポジトリでのみ build されます。フォークでは、`validate-mdx` GitHub Action がフォーク PR 内の `mint dev` と `mint broken-links` command のステータスを確認します。

**エージェント 注記**: 次を行う必要があります。

1. 既存の fork リモート があるか `git remote -v` で確認し、存在する場合は URL からユーザー名を抽出します。
2. リモート からユーザー名が見つからない場合は、`git config` で GitHub ユーザー名を確認します。
3. どちらでも見つからない場合にのみ、ユーザーに GitHub ユーザー名を確認します。
4. テストに使用できる `wandb/docs` のフォークがあることを Verify します。
5. フォークに直接 プッシュ できない場合は、ユーザーがそこから プッシュ できるように、`wandb/docs` に一時ブランチを作成します。

<div id="testing-requirements">
  ### テスト要件
</div>

ワークフローの変更をテストするには、次の作業が必要です。

1. フォークの `main` を元のリポジトリの `main` と Sync し、一時的なコミットはすべて破棄します。
2. 変更はフォークの `main` ブランチに適用します (ブランチ ブランチだけではありません) 
3. ワークフローをトリガーするため、内容の変更を含むテスト用 PR をフォークの `main` に対して作成します。

<div id="step-by-step-testing-process">
  ## step別のテスト手順
</div>

<div id="1-initial-setup">
  ### 1. 初期設定
</div>

```bash
# 既存のリモートを確認する
git remote -v

# フォークのリモートが存在する場合は、フォークURLからユーザー名を取得する
# フォークのリモートが存在しない場合は、git configでユーザー名を確認する
git config user.name  # or git config github.user

# リモートまたは設定にユーザー名が見つからない場合のみ、ユーザーにGitHubユーザー名またはフォークの詳細を確認する
# 質問例: "テストに使用するフォークのGitHubユーザー名を教えてください。"

# フォークのリモートが存在しない場合は追加する:
git remote add fork https://github.com/<username>/docs.git  # <username>を実際のユーザー名に置き換える
```


<div id="2-sync-fork-and-prepare-test-branch">
  ### 2. フォークを Sync し、テストブランチを準備する
</div>

```bash
# originから最新を取得する
git fetch origin

# mainをチェックアウトし、origin/mainにハードリセットしてクリーンな同期を確保する
git checkout main
git reset --hard origin/main

# フォークに強制プッシュして同期する（フォーク内の一時的なコミットを破棄）
git push fork main --force

# ワークフローの変更用テストブランチを作成する
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. ワークフローの変更を反映する
</div>

ワークフローファイルを変更します。依存関係をアップグレードする場合は、次の点を確認してください。

* `uses:` ステートメント内のバージョン番号を更新する
* 依存関係が複数箇所で使用されている場合は、両方のワークフローファイルを確認する

**プロのヒント**: 手順書を確定する前に、次のようなプロンプトで AI エージェントにレビューを依頼します。

> &quot;この手順書をレビューして、AI エージェントにとってより役立つものにするための改善点を提案してください。明確さ、完全性、曖昧さの排除に重点を置いてください。&quot;

<div id="5-commit-and-push-to-forks-main">
  ### 5. フォークの main にコミットしてプッシュ
</div>

```bash
# すべての変更をコミット
git add -A
git commit -m "test: [Description of what you're testing]"

# フォークのメインブランチにプッシュ
git push fork HEAD:main --force-with-lease
```

**フォークへのアクセスに関するエージェント向け手順**:
フォークに直接プッシュできない場合:

1. 変更を含む一時ブランチを wandb/docs に作成します
2. ユーザーに次のコマンドを伝えます:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. 次の URL で PR を作成するよう案内します: `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. テスト後は wandb/docs から一時ブランチを削除することを忘れないでください


<div id="6-create-test-pr">
  ### 6. テスト用のPRを作成する
</div>

```bash
# 更新されたフォークのmainから新しいブランチを作成する
git checkout -b test-pr-[description]

# ワークフローをトリガーするために小さなコンテンツ変更を加える
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# コミットしてプッシュする
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

次に、GitHub UI で `<username>:test-pr-[description]` から `<username>:main` への PR を作成します


<div id="7-monitor-and-verify">
  ### 7. 監視と確認
</div>

想定される動作:

1. GitHub Actions botが &quot;Generating preview links...&quot; という初回コメントを作成する
2. ワークフローがエラーなく完了する

確認事項:

* ✅ ワークフローが正常に完了する
* ✅ プレビューコメントが作成され、更新される
* ✅ リンクにオーバーライドURLが使用される
* ✅ ファイルの分類が正しく機能する (Added/Modified/Deleted/Renamed) 
* ❌ Actionsログにエラーがある
* ❌ セキュリティ警告やシークレットの露出がある

<div id="8-cleanup">
  ### 8. クリーンアップ
</div>

テスト後:

```bash
# フォークのmainをアップストリームに合わせてリセット
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# フォークとoriginからテストブランチを削除
git branch -D test-[description]-[date] test-pr-[description]
```


<div id="common-issues-and-solutions">
  ## よくある問題と対処法
</div>

<div id="issue-permission-denied-when-pushing-to-fork">
  ### 問題: フォークへのプッシュ時に Permission denied が発生する
</div>

* GitHubトークンが読み取り専用である可能性があります
* 解決策: SSHを使用するか、ローカル環境から手動でプッシュしてください

<div id="issue-workflows-not-triggering">
  ### 問題: ワークフローがトリガーされない
</div>

* 注意: ワークフローはPRブランチではなく、ベースブランチ (main) から実行されます
* 変更がフォークのmainブランチにあることを確認してください

<div id="issue-changed-files-not-detected">
  ### 問題: 変更されたファイルが検出されない
</div>

* コンテンツの変更が、追跡対象のディレクトリ (content/、static/、assets/ など) に含まれていることを確認してください
* ワークフローの設定にある `files:` フィルターを確認してください

<div id="testing-checklist">
  ## テストチェックリスト
</div>

* [ ] ユーザーに GitHub のユーザー名とフォークの詳細を確認した
* [ ] 2 つのリモート (origin と fork) が設定されている
* [ ] ワークフロー の変更が該当する両方のファイルに適用されている
* [ ] 変更が fork の main ブランチにプッシュされている (直接、またはユーザー経由) 
* [ ] コンテンツの変更を含むテスト用 PR を作成した
* [ ] プレビューコメントが正常に生成されている
* [ ] GitHub Actions のログにエラーがない
* [ ] テスト後に fork の main ブランチをリセットした
* [ ] wandb/docs の一時ブランチをクリーンアップした (作成した場合)