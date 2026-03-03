---
title: GitHub Actions の変更をテストする
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # エージェント用プロンプト: wandb/docs における GitHub Actions 変更のテスト
</div>

<div id="requirements">
  ## 要件
</div>

- **W&B 従業員アクセス**: 社内の W&B システムにアクセスできる W&B 従業員である必要があります。
- **GitHub フォーク**: ワークフローの変更をテストするための wandb/docs の個人用フォークが必要です。フォークでは、デフォルトブランチへの push 権限と、ブランチ保護ルールをバイパスできる権限が必要です。

<div id="agent-prerequisites">
  ## エージェントの前提条件
</div>

開始する前に、次の情報を集めてください:

1. **GitHub ユーザー名** - まず `git remote -v` でフォークのリモートを確認し、その後 `git config` でユーザー名を確認します。どちらにも見つからない場合にのみ、ユーザーに尋ねてください。
2. **フォークの状態** - デフォルトブランチに push でき、かつブランチ保護をバイパスできる権限を持つ wandb/docs のフォークがあることを確認してください。
3. **テストの範囲** - どのような変更をテストしているか（依存関係のアップグレード、機能変更など）を尋ねてください。

<div id="task-overview">
  ## タスクの概要
</div>

wandb/docs リポジトリで GitHub Actions ワークフローへの変更をテストします。

<div id="context-and-constraints">
  ## 前提と制約
</div>

<div id="repository-setup">
  ### リポジトリのセットアップ
</div>

- **メインリポジトリ**: `wandb/docs` (origin)
- **テスト用フォーク**: `<username>/docs` (fork remote) - `git remoter -v` から判別できない場合は、ユーザーにフォークのエンドポイントを確認してください。
- **重要**: PR 内の GitHub Actions は、常に PR ブランチではなくベースブランチ (main) から実行されます。
- **Mintlify デプロイの制約**: Mintlify のデプロイと `link-rot` チェックは、フォークではなくメインの wandb/docs リポジトリに対してのみビルドされます。フォークでは、`validate-mdx` GitHub Action が、フォークの PR 内で実行される `mint dev` と `mint broken-links` コマンドのステータスを確認します。

**Agent メモ**: 次を実行する必要があります:

1. 既存の fork remote を確認するために `git remote -v` を実行し、URL にユーザー名が含まれていれば抽出します。
2. remote からユーザー名が見つからない場合は、`git config` を確認して GitHub のユーザー名を探します。
3. どちらにも存在しない場合にのみ、ユーザーに GitHub ユーザー名を尋ねます。
4. テストに使用できる wandb/docs のフォークをユーザーが持っていることを確認します。
5. フォークに直接 push できない場合は、ユーザーがそこから push できるように、wandb/docs に一時ブランチを作成します。

<div id="testing-requirements">
  ### テスト要件
</div>

ワークフローの変更をテストするには、次を実行します。

1. フォーク側の `main` をメインリポジトリの `main` と同期し、一時的なコミットはすべて破棄します。
2. フォークの `main` ブランチ（フィーチャーブランチではなく）に変更を適用します。
3. ワークフローをトリガーするために、内容の変更を含むテスト用 PR をフォークの `main` に対して作成します。

<div id="step-by-step-testing-process">
  ## ステップバイステップのテストプロセス
</div>

<div id="1-initial-setup">
  ### 1. 初期設定
</div>

```bash
# 既存のリモートを確認する
git remote -v

# forkリモートが存在する場合、フォークURLからユーザー名を確認する
# forkリモートが存在しない場合、git configでユーザー名を確認する
git config user.name  # or git config github.user

# リモートまたはconfigにユーザー名が見つからない場合のみ、ユーザーにGitHubユーザー名またはフォークの詳細を確認する
# 質問例: "テストに使用するフォークのGitHubユーザー名を教えてください。"

# forkリモートが存在しない場合、追加する:
git remote add fork https://github.com/<username>/docs.git  # <username>を実際のユーザー名に置き換える
```


<div id="2-sync-fork-and-prepare-test-branch">
  ### 2. フォークを同期してテスト用ブランチを作成する
</div>

```bash
# originから最新を取得する
git fetch origin

# mainをチェックアウトし、origin/mainにハードリセットしてクリーンな同期を確保する
git checkout main
git reset --hard origin/main

# フォークに強制プッシュして同期する（フォーク内の一時的なコミットを破棄する）
git push fork main --force

# ワークフロー変更用のテストブランチを作成する
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. ワークフローの変更を適用する
</div>

ワークフローファイルに変更を加えます。依存関係をアップグレードする場合は、次を行います。

- `uses:` ステートメント内のバージョン番号を更新する
- 依存関係が複数箇所で使われている場合は、両方のワークフローファイルを確認する

**プロ向けのヒント**: 任意の runbook を確定する前に、次のようなプロンプトで AI エージェントにレビューを依頼します:

> "Please review this runbook and suggest improvements to make it more useful for AI agents. Focus on clarity, completeness, and removing ambiguity."

<div id="5-commit-and-push-to-forks-main">
  ### 5. フォーク先の main ブランチにコミットしてプッシュする
</div>

```bash
# すべての変更をコミット
git add -A
git commit -m "test: [Description of what you're testing]"

# フォークのmainブランチにプッシュ
git push fork HEAD:main --force-with-lease
```

**フォークへのアクセスに関するエージェント向け手順**:
フォークに直接 push できない場合は、次の手順に従ってください。

1. 変更を加えた一時ブランチを wandb/docs に作成する
2. ユーザーに次のコマンドを伝える:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. 次の URL で PR を作成するよう案内する: `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. テスト後に wandb/docs から一時ブランチを削除することを忘れないこと


<div id="6-create-test-pr">
  ### 6. テスト用 PR を作成
</div>

```bash
# 更新されたフォークのmainから新しいブランチを作成する
git checkout -b test-pr-[description]

# ワークフローをトリガーするために小さな変更を加える
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# コミットしてプッシュする
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

その後、GitHub の UI から `<username>:test-pr-[description]` から `<username>:main` への PR を作成します


<div id="7-monitor-and-verify">
  ### 7. モニタリングと検証
</div>

想定される挙動:

1. GitHub Actions ボットが「Generating preview links...」というメッセージの初期コメントを作成する
2. ワークフローがエラーなく完了する

次を確認する:

- ✅ ワークフローが正常に完了する
- ✅ プレビューコメントが作成され、更新される
- ✅ リンクがオーバーライド URL を使用している
- ✅ ファイルの分類が機能している (Added/Modified/Deleted/Renamed)
- ❌ Actions のログ内のエラー
- ❌ セキュリティ警告や漏洩しているシークレット

<div id="8-cleanup">
  ### 8. クリーンアップ
</div>

テストが完了したら、次を実行します：

```bash
# フォークのmainをupstreamに合わせてリセット
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# フォークとoriginからテストブランチを削除
git branch -D test-[description]-[date] test-pr-[description]
```


<div id="common-issues-and-solutions">
  ## よくある問題と対処方法
</div>

<div id="issue-permission-denied-when-pushing-to-fork">
  ### 問題: フォーク先へのプッシュ時に Permission denied エラーが発生する
</div>

- GitHub トークンが読み取り専用になっている可能性がある
- 解決策: SSH を使うか、ローカルマシンから手動でプッシュする

<div id="issue-workflows-not-triggering">
  ### 問題: Workflow がトリガーされない
</div>

- 注意: Workflow は PR ブランチではなく、ベースブランチ (main) で実行されます
- 変更が fork の main ブランチにあることを確認してください

<div id="issue-changed-files-not-detected">
  ### 問題: 変更されたファイルが検出されない
</div>

- コンテンツの変更が追跡対象のディレクトリ（content/, static/, assets/ など）内に含まれていることを確認する
- ワークフロー設定の `files:` フィルターを確認する

<div id="testing-checklist">
  ## テスト用チェックリスト
</div>

- [ ] ユーザーに GitHub のユーザー名とフォークの詳細を尋ねた
- [ ] 両方のリモート（origin と fork）が設定されている
- [ ] ワークフローの変更が両方の該当ファイルに適用されている
- [ ] 変更がフォークの main ブランチにプッシュされている（直接、またはユーザー経由）
- [ ] コンテンツの変更を含むテスト PR を作成した
- [ ] プレビューコメントが正常に生成された
- [ ] GitHub Actions のログにエラーがない
- [ ] テスト後にフォークの main ブランチをリセットした
- [ ] 一時ブランチを wandb/docs からクリーンアップした（作成していた場合）