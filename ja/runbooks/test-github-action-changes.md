---
title: GitHub Actions の変更をテストする
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # エージェント プロンプト: wandb/docs の GitHub Actions 変更のテスト
</div>

<div id="requirements">
  ## 要件
</div>

- **W&B 従業員アクセス**: W&B の従業員であり、社内の W&B システムにアクセスできる必要があります。
- **GitHub フォーク**: ワークフローの変更をテストするための wandb/docs の個人フォーク。フォーク内のデフォルトブランチに push できることと、ブランチ保護ルールをバイパスできる権限が必要です。

<div id="agent-prerequisites">
  ## エージェントの前提条件
</div>

開始する前に、次の情報を集めてください。

1. **GitHub username** - まず `git remote -v` を確認して fork の remote を確認し、その後 `git config` で username を確認します。どちらにも見つからない場合にのみユーザーに尋ねてください。
2. **Fork status** - デフォルトブランチに push でき、ブランチ保護をバイパスできる権限を持つ wandb/docs リポジトリの fork を所有していることを確認します。
3. **Test scope** - 具体的にどのような変更をテストしているか（依存関係のアップグレード、機能変更など）を尋ねます。

<div id="task-overview">
  ## タスクの概要
</div>

wandb/docs リポジトリの GitHub Actions ワークフローの変更をテストします。

<div id="context-and-constraints">
  ## コンテキストと制約
</div>

<div id="repository-setup">
  ### リポジトリのセットアップ
</div>

- **メインリポジトリ**: `wandb/docs` (origin)
- **テスト用フォーク**: `<username>/docs` (fork remote) - `git remoter -v` から明確でない場合は、ユーザーに自分のフォークのエンドポイントを尋ねてください。
- **重要**: PR の GitHub Actions は、常に PR ブランチではなくベースブランチ (main) 上で実行されます。
- **Mintlify デプロイの制限**: Mintlify のデプロイと `link-rot` チェックは、フォークではなくメインの wandb/docs リポジトリに対してのみビルドされます。フォークでは、`validate-mdx` GitHub Action が、フォーク PR における `mint dev` と `mint broken-links` コマンドのステータスをチェックします。

**Agent メモ**: 次のことを行う必要があります:

1. 既存の fork remote を確認するために `git remote -v` をチェックし、存在する場合は URL から username を抽出する。
2. remotes で username が見つからない場合は、`git config` から GitHub username を確認する。
3. どちらにも存在しない場合にのみ、ユーザーに GitHub username を尋ねる。
4. テストに使用できる wandb/docs のフォークをユーザーが持っていることを確認する。
5. フォークに直接 push できない場合は、ユーザーがそこから push できるように、wandb/docs に一時的なブランチを作成する。

<div id="testing-requirements">
  ### テスト要件
</div>

workflow の変更をテストするには、次を実施する必要があります。

1. 一時的なコミットをすべて破棄したうえで、fork 側の `main` をメインリポジトリの `main` と同期する。
2. 変更を fork の main ブランチ（feature ブランチだけでなく）に適用する。
3. workflow をトリガーするためにコンテンツの変更を含めて、fork の `main` に対するテスト用 PR を作成する。

<div id="step-by-step-testing-process">
  ## ステップごとのテスト手順
</div>

<div id="1-initial-setup">
  ### 1. 初期設定
</div>

```bash
# 既存のリモートを確認する
git remote -v

# fork リモートが存在する場合は、fork URL からユーザー名を確認する
# fork リモートが存在しない場合は、git config でユーザー名を確認する
git config user.name  # or git config github.user

# リモートまたは config にユーザー名が見つからない場合のみ、ユーザーに GitHub ユーザー名または fork の詳細を確認する
# 質問例: "テストに使用する fork の GitHub ユーザー名を教えてください。"

# fork リモートが存在しない場合は追加する:
git remote add fork https://github.com/<username>/docs.git  # <username> を実際のユーザー名に置き換える
```


<div id="2-sync-fork-and-prepare-test-branch">
  ### 2. フォークを同期し、テスト用ブランチを準備する
</div>

```bash
# origin から最新を取得する
git fetch origin

# main をチェックアウトし、origin/main にハードリセットしてクリーンな同期を確保する
git checkout main
git reset --hard origin/main

# fork に強制プッシュして同期する（fork 内の一時的なコミットを破棄する）
git push fork main --force

# ワークフロー変更用のテストブランチを作成する
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. ワークフローの変更を適用する
</div>

ワークフロー ファイルに変更を加えます。依存関係をアップグレードする場合:

- `uses:` ステートメント内のバージョン番号を更新する
- 依存関係が複数箇所で使われている場合は、両方のワークフロー ファイルを確認する

**プロのコツ**: 任意の runbook を最終確定する前に、次のようなプロンプトで AI エージェントにレビューを依頼してください:

> 「この runbook をレビューして、AI エージェントにとってより有用になるような改善点を提案してください。わかりやすさ、網羅性、曖昧さの排除に重点を置いてください。」

<div id="5-commit-and-push-to-forks-main">
  ### 5. フォークしたリポジトリの main ブランチにコミットしてプッシュする
</div>

```bash
# すべての変更をコミット
git add -A
git commit -m "test: [Description of what you're testing]"

# フォークの main ブランチにプッシュ
git push fork HEAD:main --force-with-lease
```

**fork へのアクセスに関する Agent 向け手順**:
fork に直接 push できない場合は、次の手順に従う:

1. 変更を含む一時ブランチを wandb/docs リポジトリに作成する
2. ユーザーに次のコマンドを実行するよう伝える:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. 次の URL から PR を作成してもらうよう案内する: `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. テストが完了したら、wandb/docs から一時ブランチを必ず削除する


<div id="6-create-test-pr">
  ### 6. テスト用 PR を作成する
</div>

```bash
# 更新されたフォークの main から新しいブランチを作成する
git checkout -b test-pr-[description]

# ワークフローをトリガーするために小さな変更を加える
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# コミットしてプッシュする
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

Then create a PR via the GitHub UI from `<username>:test-pr-[description]` to `<username>:main`


<div id="7-monitor-and-verify">
  ### 7. 監視と検証
</div>

期待される動作:

1. GitHub Actions ボットが「Generating preview links...」という初期コメントを作成する
2. ワークフローがエラーなく完了する

以下を確認する:

- ✅ ワークフローが正常に完了すること
- ✅ プレビューコメントが作成され、更新されること
- ✅ リンクが override URL を使用していること
- ✅ ファイルの分類（Added/Modified/Deleted/Renamed）が正しく行われていること
- ❌ Actions のログにエラーがないこと
- ❌ セキュリティ警告やシークレットが露出していないこと

<div id="8-cleanup">
  ### 8. クリーンアップ
</div>

テスト後に行う作業:

```bash
# フォークの main をアップストリームに合わせてリセット
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# フォークとオリジンからテストブランチを削除
git branch -D test-[description]-[date] test-pr-[description]
```


<div id="common-issues-and-solutions">
  ## よくある問題と対処方法
</div>

<div id="issue-permission-denied-when-pushing-to-fork">
  ### 問題: フォークへ push しようとすると Permission denied と表示される
</div>

- GitHub トークンが読み取り専用になっている可能性がある
- 解決策: SSH を使用するか、ローカル マシンから手動で push する

<div id="issue-workflows-not-triggering">
  ### 問題: ワークフローがトリガーされない
</div>

- 注意: ワークフローは PR ブランチではなく、ベース ブランチ (main) から実行されます
- 変更がフォーク先の main ブランチにあることを確認してください

<div id="issue-changed-files-not-detected">
  ### 問題: 変更されたファイルが検出されない
</div>

- コンテンツの変更が、追跡対象のディレクトリ (content/, static/, assets/ など) 内にあることを確認する
- ワークフローの設定内にある `files:` フィルターを確認する

<div id="testing-checklist">
  ## テストチェックリスト
</div>

- [ ] ユーザーから GitHub ユーザー名と fork の詳細を聞き取った
- [ ] 両方のリモート（origin と fork）が設定されている
- [ ] ワークフローの変更が両方の該当ファイルに適用されている
- [ ] 変更が fork の main ブランチに（直接またはユーザー経由で）push されている
- [ ] コンテンツ変更を含むテスト用 PR を作成した
- [ ] プレビューコメントが正常に生成された
- [ ] GitHub Actions のログにエラーがない
- [ ] テスト後に fork の main ブランチをリセットした
- [ ] 一時ブランチを wandb/docs から削除した（作成していた場合）