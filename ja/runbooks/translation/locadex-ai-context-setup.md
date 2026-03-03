---
title: Locadex AI コンテキストの設定
---

<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # エージェントプロンプト: W&B ドキュメント向け Locadex AI コンテキストの設定（韓国語／将来的に日本語）
</div>

<div id="requirements">
  ## 要件
</div>

- [ ] [General Translation Dashboard](https://dash.generaltranslation.com/)（Locadex コンソール）へのアクセス。
- [ ] Locadex/GT プロジェクトにリンクされたドキュメントリポジトリ（GitHub アプリがインストールされ、リポジトリが接続されていること）。
- [ ] 任意: [hw-wandb/wandb_docs_translation](https://github.com/hw-wandb/wandb_docs_translation) リポジトリへの読み取り権限（設定ファイルと言語辞書の確認用）。
- [ ] 任意: wandb/docs の `main` ブランチへのアクセス（`ko/`（および必要に応じて `ja/`）が存在し、用語集やロケールコンテキストを調整する際に手動翻訳と比較できること）。

<div id="agent-prerequisites">
  ## エージェントの前提条件
</div>

1. **どのロケールを設定しますか？**（例：今は韓国語のみ、後で日本語。）どの用語集の翻訳とロケールコンテキスト項目を追加するかが、ここで決まります。
2. **すでに用語集の CSV または用語リストはありますか？** ない場合は、以下のソースから作成するために runbook を使用してください。
3. **GT のプロジェクトはすでに作成済みで、リポジトリは接続されていますか？** そうでない場合は、まず [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify) のステップ 1〜6 を完了してください。

<div id="task-overview">
  ## タスク概要
</div>

このrunbookでは、(1) 既存の `wandb_docs_translation` ツールと、(2) `main` ブランチ上の手動翻訳された韓国語コンテンツ（および今後追加される日本語コンテンツ）から翻訳メモリと用語情報を取り込み、そのコンテキストを自動翻訳で利用できるように Locadex/General Translation プラットフォームを設定する方法を説明します。目的は、用語の一貫性を保ち、プロダクト名や技術用語に対して正しい「翻訳しない」動作を実現することです。

**各要素の配置場所:**

| What | Where | Notes |
|------|--------|------|
| **Glossary**（用語、定義、ロケールごとの訳語） | Locadex console → AI Context → Glossary | 用語の一貫した使用と、プロダクト/機能名に対する「翻訳しない」の制御を行います。CSV で一括アップロード可能です。 |
| **Locale Context**（言語固有の指示） | Locadex console → AI Context → Locale Context | 例: 韓国語の場合、アルファベットとハングルの間のスペースやフォーマットルールなど。 |
| **Style Controls**（トーン、想定読者、プロジェクト説明） | Locadex console → AI Context → Style Controls | プロジェクト全体に適用され、すべてのロケールで共有されます。 |
| **どのファイル/ロケールを翻訳するか** | Git → `gt.config.json` | `locales`、`defaultLocale`、`files` を指定します。Glossary やプロンプトはリポジトリには含めません。 |

まとめると、**自動翻訳の制御は Locadex console 側で行います**（Glossary、Locale Context、Style Controls）。**ファイルとロケールの設定は Git 側に保持します**（`gt.config.json`）。`gt.config.json` の任意の `dictionary` キーはアプリの UI 文字列（例: gt-next/gt-react）向けであり、ドキュメントの MDX 用語集向けではありません。ドキュメントの用語管理は console 上で行います。

<div id="context-and-constraints">
  ## コンテキストと制約
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### レガシーツール (wandb_docs_translation)
</div>

- **human_prompt.txt**: 絶対に翻訳してはならず（英語のまま保持する）W&B のプロダクト／機能名を列挙するファイル：Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models。`[**word**](link)` のようなリンク／リストのコンテキストでも同様。
- **system_prompt.txt**: 一般ルール（有効な Markdown にすること、コードブロック内ではコメントのみ翻訳すること、辞書を使用すること、リンク URL は翻訳しないこと。日本語／韓国語の場合は、アルファベットと CJK 文字の切り替え時やインライン装飾の前後にスペースを追加すること）。
- **configs/language_dicts/ko.yaml**: 「翻訳メモリ」が混在したファイル:
  - **英語のまま保持**（プロダクト／機能名）: 例 `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`。
  - **韓国語に翻訳**: 例 `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자。

したがって、このときの慣習は次のとおりです。**プロダクト／機能名（多くは先頭が大文字、または UI／リストのコンテキスト）は英語のまま**、**一般名詞としての用法はロケール辞書に従って翻訳**します。Locadex Glossary では、各ロケールについて「翻訳しない」と「X と翻訳する」の両方を反映させる必要があります。

<div id="locadexgt-platform-behavior">
  ### Locadex/GT プラットフォームの動作
</div>

- **Glossary**: 用語（原文のまま）＋任意の定義＋ロケールごとの任意の翻訳。翻訳しない場合は、そのロケールでは用語と同じ文字列を使用する（例: Term「W&B」、Translation (ko)「W&B」）。特定の訳語を指定する場合は、Translation (ko) に目的の訳語を設定する（例: 「artifact」→「아티팩트」）。
- **Locale Context**: 対象ロケールごとの自由形式の指示（例: 「ラテン文字とハングル文字の間にはスペースを入れる」）。
- **Style Controls**: プロジェクトに対する 1 つのセット（トーン、対象読者、説明）。すべてのロケールに適用される。
- AI Context を変更しても、既存コンテンツは自動的には再翻訳されない。既に翻訳済みのファイルに新しいコンテキストを適用するには、[Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate) を使用する。

<div id="step-by-step-process">
  ## 手順
</div>

<div id="1-gather-terminology-sources">
  ### 1. 用語ソースを収集する
</div>

- **`wandb_docs_translation` から**（存在する場合）:
  - `configs/human_prompt.txt` → 決して翻訳してはいけない用語のリスト。
  - `configs/language_dicts/ko.yaml`（および後で `ja.yaml`）→ 用語 → ロケールごとの訳語マップ。
- **main ブランチ上の手動翻訳から**（任意）: 英語（EN）ページと韓国語（KO）または日本語（JA）のページをいくつか比較して、プロダクト名や一般的な用語がどのように表現されているか（例: “run” vs “실행”、 “workspace” vs “워크스페이스”）を確認し、用語集エントリを追加または調整する。

**Agent メモ**: Agent が外部リポジトリを読み取れない場合でも、このリポジトリで提供されている CSV とロケールコンテキスト用テキストを人間が使用すれば、この runbook に従うことは可能です（下記の runbook および任意の CSV を参照）。

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. 用語集 CSV を作成または取得する
</div>

- このリポジトリにある韓国語用のあらかじめ用意された用語集 CSV を使用する: **runbooks/locadex-glossary-ko.csv**（下記「Glossary CSV」参照）、または次を含む CSV を新たに生成する:
  - **翻訳しない用語**: 用語ごとに 1 行。Definition（定義）は任意。`ko`（または「Translation (ko)」）は Term と同一。
  - **翻訳する用語**: 用語ごとに 1 行。Definition は任意。`ko` は望ましい韓国語の訳語。
- Locadex の「Upload Context CSV」が想定している正確なカラム名（例: `Term`, `Definition`, `ko` または `Translation (ko)`）を確認する。コンソールが別名を要求する場合は、CSV のヘッダーをそれに合わせて調整する。
- **CSV 形式（正しく解析されるために）**: ファイルが正しく解析されるよう、標準的な CSV のクォート規則を用いる。カンマをフィールドの区切り文字とし、カンマ・二重引用符・改行を含むフィールドは必ず二重引用符で囲む。引用符で囲まれたフィールド内では、内部の二重引用符は `""` のように二重にしてエスケープする。1 行につき 1 用語とし（「run, Run」のように複数のバリアントを 1 セルに入れない）、CSV をプログラムで生成・編集する場合は、CSV ライブラリを使用するか、そのようなフィールドを明示的に引用符で囲むこと。Term や Definition 内のカンマが引用符で囲まれていない場合、それらはカラムの区切りとみなされ、その行が壊れてしまう。

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. コンソールで Locadex プロジェクトを設定する
</div>

1. [General Translation Dashboard](https://dash.generaltranslation.com/) にサインインします。
2. wandb/docs リポジトリにリンクされているプロジェクトを開きます。
3. **AI Context**（または同等の機能：Glossary、Locale Context、Style Controls）にアクセスします。

<div id="4-upload-or-add-glossary-terms">
  ### 4. 用語集の用語をアップロードまたは追加する
</div>

- **オプション A**: **Upload Context CSV** を使用して、用語集（Term、Definition、ロケール列）を一括インポートします。プラットフォームが列を用語集の用語およびロケールごとの翻訳にマッピングします。
- **オプション B**: 用語を手動で追加します。Term、Definition（モデルの理解を助けるため）、さらに韓国語については翻訳を追加します（「翻訳しない」場合は term と同じ値、「翻訳する」場合は韓国語の文字列）。

少なくとも次の点を満たしてください:

- 英語のままにすべきプロダクト／機能名: W&B, Weights &amp; Biases, Artifacts, Runs, Experiments, Sweeps, Weave, Launch, Models, Reports, Datasets, Teams, Users, Workspace, Registered Models など。韓国語欄にはソースと同じ値を設定します。
- 一貫して翻訳すべき用語: 例 artifact → 아티팩트, sweep → 스윕, project → 프로젝트, workspace → 워크스페이스、ならびに `language_dicts/ko.yaml`（および今後追加する `ja.yaml`）に含まれる他のエントリ。

<div id="5-set-locale-context-for-korean">
  ### 5. 韓国語向けのロケールコンテキストを設定する
</div>

- ロケール **ko** を選択します。
- 既存のレガシー system_prompt と、韓国語ドキュメント向けのベストプラクティスを反映した指示を追加します。たとえば、次のような内容です。
  - ラテン文字と韓国語文字（ハングル、漢字を含む）を切り替えるときは、その前後にスペースを 1 つ入れます。
  - 韓国語の単語やフレーズの一部をインライン書式（太字、斜体、コード）で囲む場合、Markdown が正しくレンダリングされるように、その書式部分の前後にスペースを入れます。
  - コードブロックとリンクの URL は変更せず、必要に応じてその周囲の文章とコード内コメントのみを翻訳します。

ロケールコンテキストを保存します。

<div id="6-set-style-controls-project-wide">
  ### 6. スタイル設定を行う（プロジェクト全体）
</div>

- **プロジェクトの説明**: 例「Weights &amp; Biases (W&amp;B) のドキュメント: ML 実験のトラッキング、モデルレジストリ、LLM オペレーション向けの Weave、および関連製品。」
- **想定読者**: 開発者および ML 実務担当者。
- **トーン**: プロフェッショナルで技術的かつ明瞭に。直訳ではなく、自然で読みやすい日本語を優先する。

保存。

<div id="7-retranslate-if-needed">
  ### 7. 必要に応じて再翻訳する
</div>

- 既に自動翻訳済みのコンテンツがあり、Glossary または Locale Context を変更した場合は、新しいコンテキストが反映されるよう、影響のあるファイルに対してプラットフォームの **Retranslate** フローを使用してください。

<div id="verification-and-testing">
  ## 検証とテスト
</div>

- **Glossary**: アップロード後、Glossary タブでいくつかの用語をサンプリングして確認し（do-not-translate と翻訳済みの両方）、内容をチェックします。
- **Locale Context**: 韓国語（および後に追加する日本語）向けの手順が、正しいロケールの下に保存されていることを確認します。
- **品質**: サンプルページで翻訳を実行または手動で起動し、プロダクト名が英語のままになっていること、共通用語が Glossary と一致していること（例: artifact → アーティファクト が適切に使われていること）を確認します。

<div id="common-issues-and-solutions">
  ## よくある問題と解決策
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### 問題: CSV のアップロードが用語集にマッピングされない
</div>

- **原因**: 列名がプラットフォーム側の想定と一致していない可能性があります。
- **解決方法**: Locadex/GT のドキュメント、または UI 内の「Upload Context CSV」ヘルプで、列名（例: Term、Definition、locale code）を確認してください。CSV の列名をそれに合わせて変更し、再度アップロードしてください。

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### 問題: 英語のままにしておくべき用語が翻訳されてしまう
</div>

- **原因**: 用語が Glossary に登録されていない、または「翻訳しない」設定がされていない（対象ロケールでの翻訳が欠けている、もしくは誤っている）。
- **解決方法**: その用語を Glossary に、対象ロケールでも同じ値になるように追加する（例: “Artifacts” → ko: “Artifacts”）。さらに、それがプロダクト／機能名であるとモデルが理解できるように短い Definition を追加する。

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### 問題: 日本語（または他のロケール）では異なるルールが必要
</div>

- **原因**: ロケール固有の要件や好み（例: 敬体／常体、スペースの扱い、プロダクト名にカタカナを使うかどうかなど）。
- **解決策**: 対象ロケール（例: ja）専用の Locale Context を追加し、必要に応じて Glossary に「ja」列を追加するか、日本語向けの用語集エントリを手動で追加する。

<div id="cleanup-instructions">
  ## クリーンアップ手順
</div>

- コンソールのみでの設定の場合、docs リポジトリ内に一時的なブランチやファイルを作成する必要はありません。
- CSV を生成するために単発のスクリプトを作成した場合は、チームとしてそのスクリプトを残すと決定しない限り、コミットしないでください（AGENTS.md および単発スクリプトに関するユーザー規則を参照）。

<div id="checklist">
  ## チェックリスト
</div>

- [ ] human_prompt と language_dicts/ko.yaml（必要に応じて ja も）から用語を収集した。
- [ ] Glossary 用の CSV を作成または取得し、アップロード用のカラム名を確認した。
- [ ] Locadex コンソールにログインし、正しいプロジェクトを開いた。
- [ ] Glossary の用語（翻訳しない用語および翻訳済み用語）をアップロードまたは追加した。
- [ ] Korean（および必要に応じて後で Japanese）の Locale Context を設定した。
- [ ] Style Controls（description、audience、tone）を設定した。
- [ ] サンプル翻訳で検証し、必要に応じて既存コンテンツを再翻訳した。

<div id="glossary-csv">
  ## 用語集 CSV
</div>

このリポジトリには、韓国語用のスターター用語集 **runbooks/locadex-glossary-ko.csv** が含まれています。カラム構成は次のとおりです:

- **Term**: ドキュメント内に出現する元の（英語の）用語。
- **Definition**: 短い説明（AI の補助用。アップロード時は任意）。
- **ko**: 韓国語訳。「翻訳しない」場合は Term と同じ文字列を使い、「〜として翻訳する」場合は希望する韓国語の文字列を使います。

`configs/language_dicts/ko.yaml`（または main 上の手動で翻訳された韓国語ページ）から用語を追加するには、同じカラム構成で行を追加します。Locadex コンソール側でロケール翻訳用のカラム名が異なることを期待している場合（例: 「Translation (ko)」）、アップロード時、またはアップロード前に CSV 内で `ko` カラム名を変更してください。

<div id="csv-formatting-for-future-generation">
  ### 将来の生成に向けた CSV フォーマット
</div>

用語集の CSV を（手作業またはスクリプトで）作成・追記する際は、ファイルを有効な状態に保つため、次のルールに従ってください。

- **区切り文字**: カンマ（`,`）。フィールドがダブルクオートで囲まれていない限り、フィールド内でカンマを使用しないでください。
- **引用**: フィールドにカンマ、ダブルクオート、または改行が含まれる場合、そのフィールド全体をダブルクオート（`"`）で囲みます。整合性のために、必要に応じてすべてのフィールドを引用符で囲ってもかまいません。
- **エスケープ**: 引用符で囲まれたフィールド内では、リテラルのダブルクオートは 2 つのダブルクオート（`""`）で表現します。
- **1 行あたり 1 用語**: 各行は 1 つの用語とします。1 つのセルに複数の異なる表記を記載しないでください（例: Term 列に「run」と「artifact」をそれぞれ別の行で記載し、「run, artifact」のように 1 行にまとめないでください）。
- **ツール**: CSV をプログラムで生成する場合は、適切な CSV ライブラリ（例: Python の `csv` モジュールで `quoting=csv.QUOTE_MINIMAL` または `QUOTE_NONNUMERIC` を使用）を利用し、Term や Definition 内のカンマや引用符が正しく処理されるようにしてください。

<div id="notes">
  ## メモ
</div>

- **日本語対応は後から**: 日本語を追加する際は、`ja` 向けに Locale Context を再度設定し（例: 敬体の使用、アルファベットと日本語の間のスペース、インラインでの体裁用スペースなど）、`ja` 用の Glossary エントリも追加する（方針は同じ: do-not-translate = 原文のまま、translate-as = 望ましい日本語訳）。
- **Git 内の GT 設定**: `gt.config.json` にはすでに `locales` と `defaultLocale` が含まれている。Glossary や AI context はそこには保存されず、コンソール側にのみ存在する。
- **参考資料**: [GT Glossary](https://generaltranslation.com/docs/platform/ai-context/glossary)、[Locale Context](https://generaltranslation.com/docs/platform/ai-context/locale-context)、[Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls)、[Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify)。