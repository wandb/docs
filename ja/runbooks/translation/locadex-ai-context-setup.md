<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # エージェント プロンプト: W&B ドキュメント向け Locadex AI コンテキストの設定（韓国語、後に日本語）
</div>

<div id="requirements">
  ## 要件
</div>

- [ ] [General Translation Dashboard](https://dash.generaltranslation.com/)（Locadex コンソール）へのアクセス。
- [ ] Locadex/GT プロジェクトにリンクされた docs リポジトリ（GitHub アプリがインストールされ、リポジトリが接続されていること）。
- [ ] 任意: glossary や Locale Context を調整する際に手動翻訳と比較できるよう、`ko/`（および必要に応じて `ja/`）が存在する wandb/docs の `main` ブランチへのアクセス。

<div id="agent-prerequisites">
  ## エージェントの前提条件
</div>

1. **どのロケールを設定しますか？**（例: 今は Korean のみ。Japanese は後で。）これは、どの Glossary 翻訳と Locale Context エントリを追加するかを決定します。
2. **Glossary の CSV または用語リストは既にありますか？** ない場合は、以下のソースをもとにランブックを使って作成してください。
3. **GT プロジェクトは既に作成されており、リポジトリは接続されていますか？** まだの場合は、まず [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify) のステップ 1～6 を完了してください。

<div id="task-overview">
  ## タスク概要
</div>

このランブックでは、(1) 既存の `wandb_docs_translation` ツール群と、(2) `main` ブランチ上で手動翻訳された韓国語（および後で追加される日本語）のコンテンツから翻訳メモリと用語をどのように取得し、さらに Locadex/General Translation プラットフォームをどのように設定して自動翻訳がそのコンテキストを利用するようにするかを説明します。目的は、用語の一貫性を保ち、プロダクト名や技術用語に対する「翻訳しない」挙動を正しく機能させることです。

**各要素の所在:**

| 対象 | 場所 | メモ |
|------|--------|------|
| **Glossary**（用語、定義、ロケールごとの訳語） | Locadex コンソール → AI Context → Glossary | 用語の一貫した使用と、プロダクト/機能名に対する「翻訳しない」を実現します。CSV で一括アップロード可能。 |
| **Locale Context**（言語固有の指示） | Locadex コンソール → AI Context → Locale Context | 例: 韓国語の場合、アルファベットとハングル間のスペース、書式ルールなど。 |
| **Style Controls**（トーン、対象読者、プロジェクト説明） | Locadex コンソール → AI Context → Style Controls | プロジェクト全体に適用され、すべてのロケールに共通。 |
| **どのファイル/ロケールを翻訳するか** | Git → `gt.config.json` | `locales`、`defaultLocale`、`files` を指定。Glossary やプロンプトはリポジトリ内には置かない。 |

まとめると、**自動翻訳の制御は Locadex コンソール側**（Glossary、Locale Context、Style Controls）で行います。**ファイルとロケールの設定は Git 側**（`gt.config.json`）に保持します。`gt.config.json` のオプションの `dictionary` キーはアプリの UI 文字列用（例: gt-next/gt-react）であり、ドキュメントの MDX の Glossary 用ではありません。ドキュメントの用語管理はコンソール側で行います。

<div id="context-and-constraints">
  ## コンテキストと制約条件
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### 旧ツール (wandb_docs_translation)
</div>

- **human_prompt.txt**: 絶対に翻訳してはいけない（英語のままに保持する） W&B プロダクト/機能名を列挙する: Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models。`[**word**](link)` のようなリンク/リストの文脈でも同様。
- **system_prompt.txt**: 一般ルール (有効な markdown、コードブロック内ではコメントのみ翻訳、辞書を使用、リンク URL は翻訳しない。日本語/韓国語の場合: アルファベットと CJK 文字の切り替え時、およびインライン装飾の前後にスペースを追加)。
- **configs/language_dicts/ko.yaml**: 混在した「翻訳メモリ」:
  - **英語のまま保持**（プロダクト/機能名）: 例 `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`。
  - **韓国語に翻訳**: 例 `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자。

したがって従来の規約は、**プロダクト/機能名 (しばしば大文字始まり、または UI/リスト文脈) は英語のままにする**一方で、**普通名詞としての用法** についてはロケール辞書に従って訳す、というものだった。Locadex Glossary には、各ロケールごとに「翻訳しない」と「X に翻訳する」の両方を反映させる必要がある。

<div id="locadexgt-platform-behavior">
  ### Locadex/GT platform の動作
</div>

- **Glossary**: 用語（ソースと同じ）＋任意の Definition （定義）＋任意の ロケールごとの Translation （訳語）。「翻訳しない」の場合は、そのロケールでは Term と同じ文字列を使用します（例: Term 「W&B」、Translation (ko) 「W&B」）。「〜と訳す」の場合は、Translation (ko) に目的の訳語を設定します（例: 「artifact」 → 「아티팩트」）。
- **Locale Context**: ターゲットロケールごとの自由形式の指示（例: 「ラテン文字と韓国語の文字の間にスペースを入れる」）。
- **Style Controls**: プロジェクト全体に対する 1 セット（tone、audience、description）。すべてのロケールに適用されます。
- AI Context を変更しても、既存コンテンツは自動で再翻訳されません。既に翻訳済みのファイルに新しいコンテキストを適用するには [Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate) を使用してください。

<div id="step-by-step-process">
  ## 手順
</div>

<div id="1-gather-terminology-sources">
  ### 1. 用語ソースを収集する
</div>

- **wandb_docs_translation から**（存在する場合）:
  - `configs/human_prompt.txt` → 翻訳してはいけない用語の一覧。
  - `configs/language_dicts/ko.yaml`（および後に `ja.yaml`）→ term → locale ごとの翻訳マップ。
- **main 上の手動翻訳から**（任意）：いくつかの EN と KO（または JA）のページを比較して、プロダクト名や共通用語がどのように表記されているか（例: “run” vs “실행”、“workspace” vs “워크스페이스”）を確認し、用語集エントリを追加または調整する。

**Agent note**: エージェントが外部リポジトリを読めない場合でも、このリポジトリ内で提供されている CSV と locale コンテキストのテキストを人間が使用すれば、runbook に従うことができる（下記の runbook とオプションの CSV を参照）。

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. 用語集 CSV を作成または入手する
</div>

- このリポジトリにある韓国語用のあらかじめ用意された用語集 CSV を使用する: **runbooks/locadex-glossary-ko.csv**（下記「Glossary CSV」を参照）か、次を含む CSV を自分で生成する:
  - **翻訳禁止の用語**: 用語ごとに 1 行；Definition 列は任意；`ko`（または “Translation (ko)”）は Term と同じ値にする。
  - **翻訳対象の用語**: 用語ごとに 1 行；Definition 列は任意；`ko` に目的の韓国語訳を設定する。
- Locadex の “Upload Context CSV” が期待する正確な列名（例: `Term`、`Definition`、`ko` または `Translation (ko)`）を確認する。コンソールが異なる名前を要求する場合は CSV ヘッダーを調整する。
- **CSV フォーマット（正しくパースするために）**: ファイルが正しくパースされるよう、標準的な CSV のクオート方法を使用する。カンマはフィールド区切り文字とし、カンマ、ダブルクオート、改行のいずれかを含むフィールドは **必ず** ダブルクオートで囲む。クオートされたフィールド内では、内部のダブルクオートは `""` のように 2 つ重ねてエスケープする。1 行に含める用語は 1 つだけにする（“run, Run” のように複数のバリアントを 1 つのセルに入れない）。CSV をプログラムで生成または編集する場合は、CSV ライブラリを使用するか、そのようなフィールドを明示的にクオートする。Term や Definition にクオートされていないカンマがあると列の区切りとして扱われ、その行が壊れる。

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. コンソールで Locadex プロジェクトを設定する
</div>

1. [General Translation Dashboard](https://dash.generaltranslation.com/) にサインインします。
2. wandb/docs リポジトリにリンクされたプロジェクトを開きます。
3. **AI Context**（または同等の機能：Glossary、Locale Context、Style Controls）に移動します。

<div id="4-upload-or-add-glossary-terms">
  ### 4. 用語集の用語をアップロードまたは追加する
</div>

- **オプション A**: **Upload Context CSV** を使用して、用語集（Term、Definition、およびロケール列）を一括インポートします。プラットフォームは、列を用語集の Term とロケールごとの訳語にマッピングします。
- **オプション B**: 用語を手動で追加します。Term、Definition（モデルの理解を助けるもの）、そして Korean 用には訳語（「翻訳しない」場合は term と同じ、「翻訳する」場合は韓国語の文字列）を追加します。

少なくとも次の点を必ず設定します:

- 英語のままにしておく必要があるプロダクト／機能名: W&B、Weights & Biases、Artifacts、Runs、Experiments、Sweeps、Weave、Launch、Models、Reports、Datasets、Teams、Users、Workspace、Registered Models など。Korean = source と同じ。
- 一貫して翻訳する必要がある用語: 例 artifact → 아티팩트、sweep → 스윕、project → 프로젝트、workspace → 워크스페이스、ならびに `language_dicts/ko.yaml`（および後で `ja.yaml`）に含まれるその他のエントリ。

<div id="5-set-locale-context-for-korean">
  ### 5. Korean 用の locale context を設定する
</div>

- locale **ko** を選択します。
- 従来の system_prompt と韓国語ドキュメント向けのベストプラクティスを反映した指示を追加します。例えば次のようにします。
  - ラテン文字と韓国語の文字（ハングル、漢字）を切り替えるときは、間にスペースを入れます。
  - 韓国語の単語やフレーズの一部をインラインの装飾（太字、斜体、コード）で囲む場合は、Markdown が正しくレンダリングされるよう、その装飾部分の前後にスペースを入れます。
  - コードブロックとリンクの URL は変更せず、必要に応じて周囲の文章とコード内コメントのみ翻訳します。

locale context を保存します。

<div id="6-set-style-controls-project-wide">
  ### 6. スタイル設定を行う（プロジェクト全体）
</div>

- **プロジェクトの説明**: 例 「Weights &amp; Biases (W&amp;B) のドキュメント: ML 実験管理、モデルレジストリ、LLM Ops 向けの Weave、および関連するプロダクト。」
- **対象読者**: 開発者および ML 実務者。
- **トーン**: プロフェッショナル、技術的、明確。直訳ではなく自然で読みやすい訳を優先する。

保存します。

<div id="7-retranslate-if-needed">
  ### 7. 必要に応じて再翻訳する
</div>

- すでに自動翻訳済みのコンテンツがあり、Glossary や Locale Context を変更した場合には、新しいコンテキストが適用されるように、該当するファイルに対してプラットフォームの **Retranslate** フローを実行してください。

<div id="verification-and-testing">
  ## 検証とテスト
</div>

- **Glossary**: アップロード後、Glossary タブでいくつかの用語（「翻訳しない」指定のものと翻訳されるもの）をスポットチェックする。
- **Locale Context**: 韓国語（および後で追加される日本語）の指示が正しいロケールの下に保存されていることを確認する。
- **Quality**: サンプル ページで翻訳を実行またはトリガーし、プロダクト名が英語のままであることと、一般的な用語が Glossary と一致していることを確認する（例: artifact → 아티팩트 が適切な箇所で使われているか）。

<div id="common-issues-and-solutions">
  ## よくある問題と解決策
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### 問題: CSV のアップロードが Glossary に正しくマッピングされない
</div>

- **原因**: 列名がプラットフォームで想定されている名前と一致していない可能性があります。
- **解決方法**: Locadex/GT のドキュメントや UI 内のヘルプで、「Upload Context CSV」の列名（例: Term、Definition、locale code）を確認します。CSV の列名をそれに合わせて変更し、再度アップロードしてください。

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### Issue: 英語のままにしておくべき用語が翻訳されてしまう
</div>

- **原因**: 用語が Glossary に含まれていないか、「翻訳しない」が設定されていない（ロケール別の訳語が未設定または誤っている）。
- **解決策**: その用語を対象ロケールでも同じ値で Glossary に追加する（例: “Artifacts” → ko: “Artifacts”）。モデルがそれを product / feature 名として認識できるように、短い Definition を追加する。

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### 課題: 日本語（または他のロケール）には異なるルールが必要
</div>

- **原因**: ロケール固有の表記や文体の好み（例: 敬体の使用、スペースの入れ方、製品名のカタカナ表記など）。
- **解決策**: 対象ロケール（例: ja）用の Locale Context を別途追加し、必要に応じて「ja」列付きの Glossary エントリや、日本語向けの手動エントリを追加する。

<div id="cleanup-instructions">
  ## クリーンアップ手順
</div>

- コンソールのみでの設定の場合、docs リポジトリに一時的なブランチやファイルは不要です。
- CSV を作成するためのワンオフ スクリプトを作成した場合は、チームでそのスクリプトを残す方針にならない限りコミットしないでください（AGENTS.md と、ワンオフ スクリプトに関するユーザー向けルールを参照してください）。

<div id="checklist">
  ## チェックリスト
</div>

- [ ] human_prompt と language_dicts/ko.yaml（および必要に応じて ja）から用語を収集した。
- [ ] Glossary CSV を作成または取得し、アップロード用の列名を確認した。
- [ ] Locadex コンソールにログインし、正しいプロジェクトを開いた。
- [ ] Glossary の用語（翻訳しない用語と翻訳済み用語）をアップロードまたは追加した。
- [ ] Korean（および必要に応じて後で Japanese）向けの Locale Context を設定した。
- [ ] Style Controls（description、audience、tone）を設定した。
- [ ] サンプル翻訳で検証し、必要に応じて既存のコンテンツを再翻訳した。

<div id="glossary-csv">
  ## 用語集 CSV
</div>

このリポジトリには、韓国語用のスターター用語集が用意されています: **runbooks/locadex-glossary-ko.csv**。列は次のとおりです。

- **Term**: ドキュメント内に現れるソース (英語) 用語。
- **Definition**: 短い説明 (AI に役立つ情報。アップロード時は省略可)。
- **ko**: 韓国語訳。「翻訳しない」場合は Term と同じ文字列を使用し、「このように翻訳する」場合は目的の韓国語文字列を使用します。

`configs/language_dicts/ko.yaml` (または main 上の手動で作成した韓国語ページ) から用語を追加するには、同じ列構成で行を追加します。Locadex コンソールがロケール別翻訳用に異なる列名 (例: “Translation (ko)”) を想定している場合は、アップロード時、またはアップロード前に CSV 内で `ko` 列の名前を変更してください。

<div id="csv-formatting-for-future-generation">
  ### 将来の生成のための CSV フォーマット
</div>

用語集の CSV を作成または追記する際（手動・スクリプトどちらの場合も）、ファイルを正しい形式に保つために次のルールに従ってください。

- **区切り文字**: カンマ（`,`）。フィールド内でカンマを使用する場合は、そのフィールドを必ず二重引用符で囲んでください。
- **引用**: フィールドにカンマ、二重引用符、または改行が含まれる場合は、そのフィールド全体を二重引用符（`"`）で囲みます。一貫性のため、すべてのフィールドを引用符で囲んでもかまいません。
- **エスケープ**: 引用符で囲まれたフィールド内でリテラルの二重引用符を表すには、二重引用符 2 つ（`""`）を使用します。
- **1 行につき 1 用語**: 各行は 1 つの用語とします。1 つのセルに複数のバリエーションを記載しないでください（たとえば Term 列に「run, artifact」と書くのではなく、「run」と「artifact」を別々の行にしてください）。
- **ツール**: CSV をプログラムで生成する場合は、適切な CSV ライブラリ（例: Python の `csv` モジュールで `quoting=csv.QUOTE_MINIMAL` または `QUOTE_NONNUMERIC` を指定）を使用し、Term や Definition 内のカンマや引用符が正しく処理されるようにしてください。

<div id="notes">
  ## メモ
</div>

- **日本語は後から追加**: 日本語を追加する場合は、`ja` 用の Locale Context を再度設定する（例: 敬体、アルファベットと日本語の間のスペース、インライン書式時のスペースなど）とともに、`ja` 用の Glossary エントリを追加する（同じ方針: do-not-translate = 原文と同じ、translate-as = 望ましい日本語訳）。
- **Git における GT 設定**: `gt.config.json` にはすでに `locales` と `defaultLocale` が含まれている。Glossary や AI context はそこには保存されず、コンソール側にのみ保存される。
- **参考資料**: [GT Glossary](https://generaltranslation.com/docs/platform/ai-context/glossary)、[Locale Context](https://generaltranslation.com/docs/platform/ai-context/locale-context)、[Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls)、[Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify)。