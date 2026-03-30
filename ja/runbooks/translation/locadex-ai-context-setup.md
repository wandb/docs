<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # Agent プロンプト: W&amp;B ドキュメント向けの Locadex AI コンテキストを設定する (韓国語、後に日本語)
</div>

<div id="requirements">
  ## 要件
</div>

* [ ] [General Translation Dashboard](https://dash.generaltranslation.com/) (Locadex コンソール) へのアクセス権。
* [ ] Locadex/GT プロジェクトに紐付けられた docs リポジトリ (GitHub アプリがインストール済みで、リポジトリが接続されていること) 。
* [ ] 任意: `ko/` (必要に応じて `ja/` も) が存在する wandb/docs の `main` ブランチへのアクセス権。Glossary や Locale Context を調整する際に、手動翻訳との比較に使用します。

<div id="agent-prerequisites">
  ## Agent の事前確認事項
</div>

1. **どのロケールを設定しますか？** (例: 現時点では韓国語のみ、後で日本語を追加。) これにより、追加する Glossary の訳語と Locale Context の項目が決まります。
2. **Glossary の CSV または用語リストはすでにありますか？** ない場合は、runbook を使用して以下のソースから作成してください。
3. **GT プロジェクトはすでに作成済みで、リポジトリは接続されていますか？** まだの場合は、先に [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify) の Steps 1–6 を完了してください。

<div id="task-overview">
  ## タスク概要
</div>

このランブックでは、(1) 既存の `wandb_docs_translation` ツールと、(2) `main` にある手動翻訳済みの韓国語コンテンツ (後に日本語コンテンツも追加) から翻訳メモリと用語集を取り込み、自動翻訳がそのコンテキストを利用するように Locadex/General Translation プラットフォームを設定する方法を説明します。目的は、用語の一貫性を保ち、プロダクト名や技術用語に対して「翻訳しない」が正しく機能するようにすることです。

**配置場所:**

| 内容                                      | 場所                                            | メモ                                                              |
| --------------------------------------- | --------------------------------------------- | --------------------------------------------------------------- |
| **Glossary** (用語、定義、ロケールごとの翻訳)          | Locadex コンソール → AI Context → Glossary       | 用語を一貫して使用し、プロダクト名や機能名に「翻訳しない」を適用するために使用します。CSV で一括アップロードできます。   |
| **Locale Context** (言語固有の指示)            | Locadex コンソール → AI Context → Locale Context | 例: 韓国語では、アルファベットとハングルの間のスペースや書式ルールを指定します。                       |
| **Style Controls** (トーン、対象読者、プロジェクト説明)  | Locadex コンソール → AI Context → Style Controls | プロジェクト全体に適用され、すべてのロケールに反映されます。                                  |
| **翻訳するファイル/ロケール**                       | Git → `gt.config.json`                        | `locales`、`defaultLocale`、`files`。Glossary やプロンプトはリポジトリには含めません。 |

つまり、**自動翻訳の制御は Locadex コンソール で行います** (Glossary、Locale Context、Style Controls) 。**ファイルとロケールの設定は Git に残します** (`gt.config.json`) 。`gt.config.json` のオプションの `dictionary` キーは、ドキュメント MDX の Glossary 用ではなく、アプリ UI 文字列 (例: gt-next/gt-react) 用です。ドキュメントの用語は console で管理します。

<div id="context-and-constraints">
  ## コンテキストと制約事項
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### レガシーツール (`wandb_docs_translation`)
</div>

* **human&#95;prompt.txt**: **絶対に**翻訳してはならない W&amp;B のプロダクト名／機能名を列挙します (英語のまま保持) : Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models。`[**word**](link)` のような link/list の文脈でも同様です。
* **system&#95;prompt.txt**: 一般的なルール (有効な markdown、コードブロック内ではコメントのみ翻訳、辞書を使用、リンク URL は翻訳しない、日本語／韓国語ではアルファベットと CJK 文字の切り替わり時、およびインライン書式の前後にスペースを追加) 。
* **configs/language&#95;dicts/ko.yaml**: 「翻訳メモリ」が混在:
  * **英語のまま保持** (プロダクト名／機能名) : 例 `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`。
  * **韓国語に翻訳**: 例 `artifact` → アーティファクト, `sweep` → スイープ, `project` → プロジェクト, `workspace` → ワークスペース, `user` → ユーザー。

つまり、従来の慣例は次のとおりです: **プロダクト名／機能名 (多くは先頭大文字、または UI/list の文脈) は英語のまま**、**普通名詞としての用法**はロケール辞書に従います。Locadex Glossary には、ロケールごとに「翻訳しない」と「X と翻訳する」の両方を反映させる必要があります。

<div id="locadexgt-platform-behavior">
  ### Locadex/GT プラットフォームの挙動
</div>

* **Glossary**: Term (ソース内の表記どおり) + 任意の Definition + ロケールごとの任意の翻訳。「翻訳しない」場合は、そのロケールでは用語と同じ文字列を使用します (例: Term “W&amp;B”, Translation (ko) “W&amp;B”) 。「〜として翻訳する」場合は、Translation (ko) に目的の訳語を設定します (例: “artifact” → “아티팩트”) 。
* **Locale Context**: 対象ロケールごとの自由形式の指示 (例: “Use space between Latin and Korean characters”) 。
* **Style Controls**: プロジェクト全体で 1 セット (トーン、対象読者、説明) 。すべてのロケールに適用されます。
* AI Context を変更しても、既存のコンテンツは**自動では**再翻訳されません。すでに翻訳済みのファイルに新しいコンテキストを適用するには、[Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate) を使用してください。

<div id="step-by-step-process">
  ## 手順
</div>

<div id="1-gather-terminology-sources">
  ### 1. 用語ソースを収集する
</div>

* **wandb&#95;docs&#95;translation から** (利用可能な場合) :
  * `configs/human_prompt.txt` → 絶対に翻訳してはいけない用語のリスト。
  * `configs/language_dicts/ko.yaml` (および後で `ja.yaml`)  → 用語と各ロケールでの訳語の対応表。
* **main 上の手動翻訳から** (任意) : EN と KO (または JA) のページをいくつか比較し、プロダクト名や一般的な用語がどのように訳されているか (例: “run” と “실행”、“workspace” と “워크스페이스”) を確認して、Glossary の項目を追加または調整します。

**Agent 注**: agent が外部リポジトリを読み取れない場合でも、このリポジトリ内で提供されている CSV と locale-context テキストを使って、人手で runbook に従うことは可能です (runbooks と、以下の任意の CSV を参照) 。

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. Glossary CSV を作成または入手する
</div>

* このリポジトリにある韓国語向けの事前作成済み Glossary CSV を使用します: **runbooks/locadex-glossary-ko.csv** (以下の「Glossary CSV」を参照) 。または、次を含むものを生成します。
  * **翻訳しない用語**: 1 用語につき 1 行。Definition は任意。`ko` (または `Translation (ko)`) = Term と同じ値。
  * **翻訳済みの用語**: 1 用語につき 1 行。Definition は任意。`ko` = 希望する韓国語の訳語。
* Locadex の「Upload Context CSV」で必要とされる正確な列名を確認します (例: `Term`、`Definition`、`ko` または `Translation (ko)`) 。コンソールが別の名前を想定している場合は、CSV ヘッダーを調整します。
* **CSV 形式 (正しく解析するため)&#x20;**: ファイルが正しく解析されるよう、標準的な CSV のクォート規則を使用します。カンマはフィールド区切り文字です。カンマ、ダブルクォート、または改行を含むフィールドは、**必ず**ダブルクォートで囲む必要があります。クォートされたフィールド内では、内部のダブルクォートを二重にしてエスケープします (`""`) 。1 行につき 1 つの用語のみを記載します (「run, Run」のように複数の表記ゆれを 1 つのセルに入れないでください) 。CSV をプログラムで生成または編集する場合は、CSV ライブラリを使用するか、そのようなフィールドを明示的にクォートしてください。Term または Definition 内のクォートされていないカンマは列の区切りとして扱われ、行が壊れます。

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. コンソールでLocadex プロジェクトを設定する
</div>

1. [General Translation Dashboard](https://dash.generaltranslation.com/)にサインインします。
2. wandb/docs repository にリンクされているプロジェクトを開きます。
3. **AI Context** (または同等の項目である Glossary、Locale Context、Style Controls) に移動します。

<div id="4-upload-or-add-glossary-terms">
  ### 4. Glossary 用語をアップロードまたは追加する
</div>

* **Option A**: **Upload Context CSV** を使用して、Glossary を一括インポートします (Term、Definition、ロケール列) 。プラットフォームは各列を Glossary 用語およびロケールごとの翻訳にマッピングします。
* **Option B**: 用語を手動で追加します。追加する項目は Term、Definition (モデルの参考になります) 、そして韓国語については翻訳です (「翻訳しない」の場合は Term と同じ、「〜として翻訳する」の場合は韓国語の文字列) 。

少なくとも、次の点を確認してください。

* 英語のままにする必要があるプロダクト名 / 機能名: W&amp;B, Weights &amp; Biases, Artifacts, Runs, Experiments, Sweeps, Weave, Launch, Models, Reports, Datasets, Teams, Users, Workspace, Registered Models など。韓国語はソースと同じにします。
* 一貫して翻訳する必要がある用語: 例: artifact → アーティファクト、sweep → スイープ、project → プロジェクト、workspace → ワークスペース、および `language_dicts/ko.yaml` (後で `ja.yaml` も) のその他のエントリ。

<div id="5-set-locale-context-for-korean">
  ### 5. 韓国語のLocale Contextを設定する
</div>

* ロケール **ko** をSelectします。
* 従来の system&#95;prompt と、韓国語ドキュメントのベストプラクティスを反映した指示を追加します。たとえば、次のような内容です。
  * ラテン文字と韓国語の文字 (ハングル、漢字を含む) を切り替える際は、間にスペースを入れます。
  * 韓国語の単語や語句の一部にインライン書式 (太字、斜体、コード) を使用する場合は、Markdown が正しく表示されるよう、書式を適用した部分の前後にスペースを入れます。
  * コードブロックとリンク URL は変更せず、周囲の本文と、必要に応じてコード内のコメントのみを翻訳します。

Locale ContextをSaveします。

<div id="6-set-style-controls-project-wide">
  ### 6. Style Controls を行う (プロジェクト全体)
</div>

* **プロジェクトの説明**: 例: 「Weights &amp; Biases (W&amp;B) に関するドキュメント: ML の実験管理、モデルレジストリ、LLM ops 向けの Weave、および関連プロダクト。」
* **対象読者**: 開発者および ML の実務者。
* **トーン**: プロフェッショナルで技術的、かつ明快。直訳よりも自然で読みやすい表現を優先します。

保存。

<div id="7-retranslate-if-needed">
  ### 7. 必要に応じて再翻訳する
</div>

* すでに自動翻訳済みのコンテンツがあり、Glossary または Locale Context を変更した場合は、新しいコンテキストが適用されるよう、影響を受けるファイルに対してプラットフォームの **Retranslate** フローを実行してください。

<div id="verification-and-testing">
  ## 検証とテスト
</div>

* **Glossary**: アップロード後、Glossary タブでいくつかの用語 (未翻訳のものと翻訳済みのもの) を抜き取り確認します。
* **Locale Context**: 韓国語 (および後で日本語) の指示が、正しいロケールの下に保存されていることを確認します。
* **Quality**: サンプルページで翻訳を実行または trigger し、プロダクト名が英語のままであること、および一般的な用語が Glossary と一致していることを確認します (例: artifact → 아티팩트 (適切な場合) ) 。

<div id="common-issues-and-solutions">
  ## よくある問題と解決策
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### 問題: CSV upload が Glossary にマッピングされない
</div>

* **原因**: 列名がプラットフォームで想定されているものと一致していない可能性があります。
* **解決策**: 「Upload Context CSV」の列名 (例: Term、Definition、locale code) については、Locadex/GT のドキュメントまたは UI 内のヘルプを確認してください。CSV の列名を修正して、再度アップロードしてください。

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### 問題: 英語のままにすべき用語まで翻訳されてしまう
</div>

* **原因**: 用語がGlossaryに登録されていない、または「翻訳しない」が設定されていない (対象ロケールの翻訳が未設定、または誤っている) 。
* **解決策**: 対象ロケールでも同じ値になるよう、その用語をGlossaryに追加します (例: “Artifacts” → ko: “Artifacts”) 。あわせて短いDefinitionも追加し、それがプロダクト名または機能名であることをモデルが理解できるようにします。

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### 問題: 日本語 (または他のロケール) には別のルールが必要
</div>

* **原因**: ロケール固有の好み (例: 丁寧体、スペースの扱い、プロダクト名のカタカナ表記) 。
* **解決策**: そのロケール用に別の Locale Context (例: ja) を追加し、必要に応じて「ja」列を含む追加の Glossary エントリ、または日本語向けの手動エントリを追加します。

<div id="cleanup-instructions">
  ## クリーンアップ手順
</div>

* コンソールのみで設定する場合、docs リポジトリに一時的なブランチやファイルは不要です。
* CSV を作成するために一時的なスクリプトを作成した場合は、チームが保持すると判断しない限り、コミットしないでください (単発スクリプトに関する `AGENTS.md` とユーザールールを参照) 。

<div id="checklist">
  ## チェックリスト
</div>

* [ ] human&#95;prompt、language&#95;dicts/ko.yaml (該当する場合は ja も) から用語を収集した。
* [ ] アップロード用の Glossary CSV を作成または入手し、列名を確認した。
* [ ] Locadex コンソールにログインし、正しいプロジェクトを開いた。
* [ ] Glossary 用語 (翻訳しない語句と翻訳済みの語句) をアップロードまたは追加した。
* [ ] 韓国語の Locale Context を設定した (必要に応じて後で日本語も) 。
* [ ] Style Controls (説明、対象読者、トーン) を設定した。
* [ ] サンプル翻訳で確認し、必要に応じて既存コンテンツを再翻訳した。

<div id="glossary-csv">
  ## Glossary CSV
</div>

このリポジトリには、韓国語のスターター用 Glossary が含まれています: **runbooks/locadex-glossary-ko.csv**。列は次のとおりです。

* **Term**: ドキュメント内で使われているソース (英語) の用語。
* **Definition**: 短い説明 (AI の補助用。upload では省略可) 。
* **ko**: 韓国語訳。「翻訳しない」場合は Term と同じ文字列を使用し、「このように翻訳する」場合は使用したい韓国語の文字列を指定します。

`configs/language_dicts/ko.yaml` (または main 上の手動 KO ページ) からさらに用語を追加するには、同じ列で行を追記します。Locadex コンソールでロケール翻訳用に別の列名 (例: “Translation (ko)”) が必要な場合は、upload 時、または upload 前に CSV 内で `ko` 列名を変更してください。

<div id="csv-formatting-for-future-generation">
  ### 今後の生成に向けたCSVのフォーマット
</div>

用語集CSVを作成または追記する際は (手作業でもスクリプトでも) 、ファイルを有効な状態に保つため、次のルールに従ってください。

* **Delimiter**: カンマ (`,`) を使用します。フィールドを引用符で囲んでいない限り、フィールド内にカンマを含めないでください。
* **Quoting**: フィールドにカンマ、二重引用符、または改行が含まれる場合は、そのフィールド全体を二重引用符 (`"`) で囲んでください。一貫性のために、すべてのフィールドを引用符で囲んでもかまいません。
* **Escaping**: 引用符で囲まれたフィールド内では、二重引用符そのものは二重引用符2つ (`""`) で表します。
* **One term per row**: 1行につき1つの用語を記載します。1つのセルに複数の表記ゆれを並べないでください (たとえば、Term 列に “run, artifact” と記載するのではなく、“run” と “artifact” はそれぞれ別の行にしてください) 。
* **Tools**: プログラムでCSVを生成する場合は、適切なCSVライブラリを使用してください (例: Python の `csv` モジュールで `quoting=csv.QUOTE_MINIMAL` または `QUOTE_NONNUMERIC` を使用) 。これにより、Term や Definition 内のカンマや引用符を正しく処理できます。

<div id="notes">
  ## メモ
</div>

* **日本語対応は後で**: 日本語を追加する際は、`ja` の Locale Context を改めて記載し (例: 丁寧体、アルファベットと日本語の間のスペース、インライン書式内のスペース) 、`ja` の Glossary エントリも追加します (考え方は同じで、do-not-translate = ソースと同じ、translate-as = 意図する日本語) 。
* **Git 内の GT 設定**: `gt.config.json` にはすでに `locales` と `defaultLocale` があります。Glossary や AI context はそこには保存されず、コンソールにのみ保存されます。
* **参考資料**: [GT Glossary](https://generaltranslation.com/docs/platform/ai-context/glossary), [Locale Context](https://generaltranslation.com/docs/platform/ai-context/locale-context), [Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls), [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).