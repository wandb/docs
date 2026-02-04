# Agent prompt: wandb/docs における GitHub Actions の変更テスト

## 必要条件
- **W&B 従業員アクセス**: 内部の W&B システムにアクセスできる W&B 従業員である必要があります。
- **GitHub のフォーク**: ワークフローの変更をテストするための wandb/docs の個人用フォーク。フォーク先で、デフォルトブランチへのプッシュおよびブランチ保護ルールのバイパス権限が必要です。

## エージェントの前提条件
開始前に、以下の情報を収集してください：
1. **GitHub ユーザー名** - まず `git remote -v` でフォークのリモートを確認し、次に `git config` でユーザー名を確認します。どちらの場所にも見当たらない場合のみ、ユーザーに尋ねてください。
2. **フォークの状態** - wandb/docs のフォークを所有しており、デフォルトブランチへのプッシュおよびブランチ保護のバイパス権限があることを確認してください。
3. **テスト範囲** - 具体的にどのような変更（依存関係のアップグレード、機能変更など）をテストしているかを尋ねてください。

## タスクの概要
wandb/docs リポジトリにおける GitHub Actions ワークフローの変更をテストします。

## コンテキストと制約

### リポジトリのセットアップ
- **メインリポジトリ**: `wandb/docs` (origin)
- **テスト用フォーク**: `<username>/docs` (fork リモート) - `git remote -v` から判断できない場合は、ユーザーにフォークのエンドポイントを尋ねてください。
- **重要**: PR における GitHub Actions は、常に PR ブランチからではなく、ベースブランチ (main) から実行されます。
- **Mintlify デプロイの制限**: Mintlify のデプロイと `link-rot` チェックは、メインの wandb/docs リポジトリに対してのみビルドされ、フォークに対しては行われません。フォークでは、`validate-mdx` GitHub Action がフォークの PR において `mint dev` と `mint broken-links` コマンドの状態をチェックします。

**エージェントへの注記**: 以下の手順を行ってください：
1. `git remote -v` をチェックして既存のフォークリモートを確認し、URL が存在すればそこからユーザー名を抽出します。
2. リモートにユーザー名が見つからない場合は、`git config` で GitHub ユーザー名を確認します。
3. どちらの場所にも見当たらない場合のみ、ユーザーに GitHub ユーザー名を尋ねます。
4. テストに使用できる wandb/docs のフォークがあることを確認します。
5. フォークに直接プッシュできない場合は、wandb/docs に一時的なブランチを作成し、ユーザーがそこからプッシュできるようにします。

### テストの要件
ワークフローの変更をテストするには、以下を行う必要があります：
1. フォークの `main` をメインリポジトリの `main` と同期させ、すべての一時的なコミットを破棄します。
2. 変更を（単なる機能ブランチではなく）フォークの main ブランチに適用します。
3. ワークフローをトリガーするために、内容の変更を含むテスト PR をフォークの `main` に対して作成します。

## ステップバイステップのテストプロセス

### 1. 初期セットアップ
```bash
# 既存のリモートを確認
git remote -v

# フォークのリモートが存在する場合、フォークの URL からユーザー名を控える
# フォークのリモートがない場合、git config でユーザー名を確認する
git config user.name  # または git config github.user

# リモートや設定で見つからない場合のみ、ユーザーに GitHub ユーザー名やフォークの詳細を尋ねる
# 質問例: "テストに使用するフォークの GitHub ユーザー名は何ですか？"

# フォークのリモートがない場合は追加する:
git remote add fork https://github.com/<username>/docs.git  # <username> を実際のユーザー名に置き換える
```

### 2. フォークの同期とテストブランチの準備
```bash
# origin から最新情報を取得
git fetch origin

# main をチェックアウトし、origin/main にハードリセットしてクリーンな同期を確保する
git checkout main
git reset --hard origin/main

# フォークに強制プッシュして同期する（フォーク内の一時的なコミットをすべて破棄）
git push fork main --force

# ワークフロー変更用のテストブランチを作成
git checkout -b test-[description]-[date]
```

### 3. ワークフロー変更の適用
ワークフローファイルに変更を加えます。依存関係のアップグレードの場合：
- `uses:` ステートメントのバージョン番号を更新します。
- 依存関係が複数の場所で使用されている場合は、両方のワークフローファイルを確認してください。

**プロのヒント**: ランブックを確定させる前に、以下のようなプロンプトで AI エージェントにレビューを依頼してください：
> 「このランブックをレビューし、AI エージェントにとってより有用になるよう改善案を提案してください。明快さ、完全性、および曖昧さの排除に焦点を当ててください。」

### 5. フォークの main へのコミットとプッシュ
```bash
# すべての変更をコミット
git add -A
git commit -m "test: [テスト内容の説明]"

# フォークの main ブランチにプッシュ
git push fork HEAD:main --force-with-lease
```

**フォークへのアクセスに関するエージェントへの指示**:
フォークに直接プッシュできない場合：
1. 変更を加えた一時的なブランチを wandb/docs に作成します。
2. ユーザーに以下のコマンドを提供します：
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. `https://github.com/<username>/docs/compare/main...test-pr-[description]` で PR を作成するようガイドします。
4. テスト終了後、wandb/docs から一時的なブランチを削除することを忘れないでください。

### 6. テスト PR の作成
```bash
# 更新されたフォークの main から新しいブランチを作成
git checkout -b test-pr-[description]

# ワークフローをトリガーするために、コンテンツに小さな変更を加える
echo "
" >> content/en/guides/quickstart.md

# コミットしてプッシュ
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

その後、GitHub UI を通じて `<username>:test-pr-[description]` から `<username>:main` への PR を作成します。

### 7. 監視と検証

期待される振る舞い:
1. GitHub Actions ボットが "Generating preview links..." という最初のコメントを作成します。
2. ワークフローがエラーなしで完了する必要があります。

以下を確認してください：
- ✅ ワークフローが正常に完了したか
- ✅ プレビューコメントが作成および更新されたか
- ✅ リンクがオーバーライド URL を使用しているか
- ✅ ファイルの分類（Added/Modified/Deleted/Renamed）が機能しているか
- ❌ Actions ログにエラーがないか
- ❌ セキュリティ警告や公開されたシークレットがないか

### 8. クリーンアップ
テスト終了後：
```bash
# フォークの main をアップストリームに合わせてリセット
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# フォークと origin からテストブランチを削除
git branch -D test-[description]-[date] test-pr-[description]
```

## よくある問題と解決策

### 問題: フォークへのプッシュ時に Permission denied が発生する
- GitHub トークンが読み取り専用である可能性があります。
- 解決策: SSH を使用するか、ローカルマシンから手動でプッシュしてください。

### 問題: ワークフローがトリガーされない
- 注意: ワークフローは PR ブランチではなく、ベースブランチ (main) から実行されます。
- 変更がフォークの main ブランチに含まれていることを確認してください。

### 問題: 変更されたファイルが検出されない
- コンテンツの変更が追跡対象のディレクトリ（content/, static/, assets/ など）にあることを確認してください。
- ワークフロー設定の `files:` フィルタを確認してください。

## テストチェックリスト

- [ ] ユーザーに GitHub ユーザー名とフォークの詳細を尋ねた
- [ ] 両方のリモート（origin と fork）が設定されている
- [ ] ワークフローの変更が関連する両方のファイルに適用されている
- [ ] 変更が（直接またはユーザーを通じて）フォークの main ブランチにプッシュされている
- [ ] コンテンツ変更を含むテスト PR が作成されている
- [ ] プレビューコメントが正常に生成されている
- [ ] GitHub Actions ログにエラーがない
- [ ] テスト後にフォークの main ブランチがリセットされている
- [ ] wandb/docs から一時的なブランチがクリーンアップされている（作成した場合）