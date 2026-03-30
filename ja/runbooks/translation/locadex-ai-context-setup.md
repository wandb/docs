<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # エージェント プロンプト: W&amp;B ドキュメント向けに Locadex AI のコンテキストを設定する (韓国語、のちに日本語)
</div>

<div id="requirements">
  ## 要件
</div>

* [ ] [General Translation Dashboard](https://dash.generaltranslation.com/) (Locadex コンソール) へのアクセス。
* [ ] Locadex/GT プロジェクトにリンクされた docs リポジトリ (GitHub アプリがインストール済みで、リポジトリが接続されていること) 。
* [ ] 任意: `ko/` (必要に応じて `ja/` も) が存在する wandb/docs の `main` ブランチへのアクセス。Glossary や Locale Context を調整する際に、手動翻訳を比較するために使用します。

<div id="agent-prerequisites">
  ## エージェントの前提条件
</div>

1. **どのロケールを設定しますか？**  (例: 現時点では韓国語のみ、日本語は後で。) 追加する Glossary の翻訳と Locale Context のエントリは、これによって決まります。
2. **Glossary の CSV または用語リストはすでにありますか？** ない場合は、runbook を使用して以下のソースを基に作成してください。
3. **GT プロジェクトはすでに作成されており、リポジトリは接続済みですか？** まだの場合は、まず [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify) の手順 1～6 を完了してください。

<div id="task-overview">
  ## タスクの概要
</div>

このランブックでは、(1) 従来の wandb&#95;docs&#95;translation ツールと、(2) `main` 上で手動翻訳された韓国語コンテンツ (および今後の日本語コンテンツ) から翻訳メモリと用語を取得する方法、および自動翻訳がそのコンテキストを使用するように Locadex/General Translation プラットフォームを設定する方法について説明します。目的は、用語の一貫性を確保し、プロダクト名や技術用語に対して正しく「翻訳しない」動作をさせることです。

**保存場所:**

| 内容                                      | 場所                                            | メモ                                                               |
| --------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------- |
| **Glossary** (用語、定義、ロケールごとの翻訳)          | Locadex コンソール → AI Context → Glossary       | 用語の一貫した使用と、プロダクト名や機能名に対する「翻訳しない」設定を制御します。CSV による一括アップロードも可能です。   |
| **Locale Context** (言語固有の指示)            | Locadex コンソール → AI Context → Locale Context | 例: 韓国語では、アルファベットとハングルの間のスペースや書式設定ルールを指定します。                      |
| **Style Controls** (トーン、対象読者、プロジェクト説明)  | Locadex コンソール → AI Context → Style Controls | プロジェクト全体に適用され、すべてのロケールに反映されます。                                   |
| **翻訳対象のファイル/ロケール**                      | Git → `gt.config.json`                        | `locales`、`defaultLocale`、`files`。glossary やプロンプトはリポジトリには含まれません。 |

したがって、**自動翻訳の制御は Locadex コンソール で行います** (Glossary、Locale Context、Style Controls) 。**ファイルとロケールの設定は Git に保持します** (`gt.config.json`) 。`gt.config.json` の省略可能な `dictionary` キーは、ドキュメント MDX の glossary 用ではなく、アプリ UI 文字列 (例: gt-next/gt-react) 向けです。ドキュメントの用語は console で管理します。

<div id="context-and-constraints">
  ## コンテキストと制約条件
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### 従来のツール群 (wandb_docs_translation)
</div>

* **human&#95;prompt.txt**: **絶対に**翻訳してはならない W&amp;B のプロダクト名 / 機能名を列挙します (英語のまま保持) : Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models。`[**word**](link)` のような link/list コンテキストでも同様です。
* **system&#95;prompt.txt**: 一般的なルール (有効な markdown、コードブロック内ではコメントのみ翻訳、辞書を使用する、リンク URL は翻訳しない、日本語 / 韓国語ではアルファベットと CJK 文字が切り替わる箇所、およびインライン書式の前後にスペースを追加) 。
* **configs/language&#95;dicts/ko.yaml**: 「翻訳メモリ」が混在しています:
  * **英語のまま保持** (プロダクト名 / 機能名) : 例: `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`。
  * **韓国語に翻訳**: 例: `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자。

したがって、この慣例は次のとおりでした: **プロダクト名 / 機能名 (多くの場合は先頭が大文字、または UI / list コンテキスト内) は英語のまま**、**普通名詞としての用法**はロケール辞書に従います。Locadex Glossary は、各ロケールについて「翻訳しない」と「X として翻訳する」の両方を反映する必要があります。

<div id="locadexgt-platform-behavior">
  ### Locadex/GT プラットフォームの動作
</div>

* **Glossary**: 用語 (ソースでの表記) + optional の Definition + ロケールごとの optional の翻訳です。「翻訳しない」場合は、そのロケールの翻訳に用語と同じ文字列を使用します (例: 用語「W&amp;B」、翻訳 (ko) 「W&amp;B」) 。「〜と翻訳する」場合は、翻訳 (ko) に目的の訳語を設定します (例: 「artifact」→「아티팩트」) 。
* **Locale Context**: ターゲットロケールごとの自由記述形式の指示です (例: 「ラテン文字と韓国語の文字の間にスペースを入れる」) 。
* **Style Controls**: プロジェクトごとに 1 セットあります (トーン、対象読者、説明) 。すべてのロケールに適用されます。
* AI Context を変更しても、既存のコンテンツは自動的には再翻訳されません。すでに翻訳済みのファイルに新しいコンテキストを適用するには、[Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate) を使用してください。

<div id="step-by-step-process">
  ## Stepごとのプロセス
</div>

<div id="1-gather-terminology-sources">
  ### 1. 用語ソースを収集する
</div>

* **wandb&#95;docs&#95;translation から** (利用可能な場合) :
  * `configs/human_prompt.txt` → 絶対に翻訳してはいけない用語のリスト。
  * `configs/language_dicts/ko.yaml` (および後で `ja.yaml`) → 用語 → ロケールごとの翻訳マップ。
* **main 上の manual translations から** (optional) : EN と KO (または JA) のページをいくつか比較し、プロダクト名や一般的な用語がどのように訳されているか (例: 「run」か「실행」、「workspace」か「워크스페이스」) を確認して、Glossary エントリを追加または調整します。

**エージェント注**: エージェントが外部リポジトリを参照できない場合でも、このリポジトリ内で提供されている CSV と locale-context テキストを使って、人手で runbook に従うことは可能です (runbooks と下記の optional CSV を参照してください) 。

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. Glossary CSV を作成するか入手する
</div>

* このリポジトリ内の韓国語用の事前作成済み Glossary CSV を使用します: **runbooks/locadex-glossary-ko.csv** (以下の「Glossary CSV」を参照) 。または、次を含むものを生成します:
  * **翻訳しない用語**: 用語ごとに 1 行。Definition は任意です。`ko` (または「Translation (ko)」) = Term と同じです。
  * **翻訳済み用語**: 用語ごとに 1 行。Definition は任意です。`ko` = 希望する韓国語の対応語です。
* Locadex の「Upload Context CSV」で想定される正確な列名 (例: `Term`、`Definition`、`ko` または `Translation (ko)`) を確認します。コンソールが異なる名前を想定している場合は、CSV ヘッダーを調整します。
* **CSV 形式 (正しく解析するため)&#x20;**: ファイルが正しく解析されるように、標準的な CSV の引用ルールを使用します。カンマはフィールド区切り文字です。カンマ、ダブルクォート、または改行を含むフィールドは、**必ず**ダブルクォートで囲む必要があります。引用符で囲まれたフィールド内では、内部のダブルクォートを 2 つ重ねてエスケープします (`""`) 。1 行につき 1 つの用語のみを記載します (「run, Run」のように複数の表記ゆれを 1 つのセルに入れないでください) 。プログラムで CSV を生成または編集する場合は、CSV ライブラリを使用するか、そのようなフィールドを明示的に引用符で囲んでください。Term または Definition に引用符で囲まれていないカンマがあると、列の区切りとして扱われ、行が壊れます。

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. コンソールで Locadex プロジェクトを設定します
</div>

1. [General Translation Dashboard](https://dash.generaltranslation.com/) にサインインします。
2. wandb/docs リポジトリにリンクされているプロジェクトを開きます。
3. **AI Context** に移動します (または同等の項目: Glossary、Locale Context、Style Controls) 。

<div id="4-upload-or-add-glossary-terms">
  ### 4. Glossary 用語をアップロードまたは追加する
</div>

* **Option A**: **Upload Context CSV** を使用して、Glossary (Term、Definition、locale 列) を一括で import します。プラットフォームは各列を Glossary 用語とロケールごとの翻訳にマッピングします。
* **Option B**: 用語を手動で追加します。Term、Definition (モデルの助けになります) 、そして Korean では翻訳を追加します (「翻訳しない」場合は Term と同じ、「～と翻訳する」場合は Korean の文字列) 。

少なくとも次を確認してください。

* 英語のままにする必要があるプロダクト名/機能名: W&amp;B、Weights &amp; Biases、Artifacts、Runs、Experiments、Sweeps、Weave、Launch、Models、Reports、Datasets、Teams、Users、Workspace、Registered Models など。Korean はソースと同じにします。
* 一貫して翻訳する必要がある用語: 例: artifact → 아티팩트、sweep → 스윕、project → 프로젝트、workspace → 워크스페이스、および `language_dicts/ko.yaml` (および後で `ja.yaml`) 内のその他のエントリ。

<div id="5-set-locale-context-for-korean">
  ### 5. 韓国語の Locale Context を設定する
</div>

* ロケール **ko** を Select します。
* 従来の `system_prompt` と韓国語ドキュメントのベストプラクティスを反映した指示を追加します。例えば、次のような内容です。
  * ラテン文字と韓国語文字 (ハングル、漢字を含む) を切り替える際は、間にスペースを入れます。
  * 韓国語の単語または語句の一部にインライン書式 (太字、斜体、コード) を使用する場合は、markdown が正しく表示されるよう、書式を適用した部分の前後にスペースを入れます。
  * コードブロックとリンク URL は変更せず、周囲の説明文と、必要に応じてコード内のコメントのみを翻訳します。

Locale Context を Save します。

<div id="6-set-style-controls-project-wide">
  ### 6. Style Controls を行う (プロジェクト全体)
</div>

* **プロジェクトの説明**: 例: 「Weights &amp; Biases (W&amp;B) のドキュメント: ML の実験管理、モデル Registry、LLM Ops 向けの Weave、および関連プロダクト。」
* **対象読者**: 開発者と ML の実務者。
* **トーン**: プロフェッショナル、技術的、明快。逐語訳よりも自然な読みやすさを優先します。

Save.

<div id="7-retranslate-if-needed">
  ### 7. 必要に応じて再翻訳する
</div>

* すでに自動翻訳されたコンテンツがあり、Glossary または Locale Context を変更した場合は、影響を受けるファイルに対してプラットフォームの **Retranslate** フローを使用して、新しいコンテキストが適用されるようにしてください。

<div id="verification-and-testing">
  ## 検証とテスト
</div>

* **Glossary**: アップロード後、Glossary タブでいくつかの用語 (非翻訳対象と翻訳済み) を抜き取り確認します。
* **Locale Context**: 韓国語 (および後で日本語) の指示が正しいロケールに保存されていることを確認します。
* **Quality**: サンプルページで翻訳を実行またはトリガーし、プロダクト名が英語のままになっていること、および一般的な用語がGlossaryと一致していることを確認します (例: artifact → 아티팩트 (適切な場合) ) 。

<div id="common-issues-and-solutions">
  ## よくある問題と解決策
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### 問題: CSV upload が Glossary にマッピングされない
</div>

* **原因**: 列名が、プラットフォームで想定されているものと一致していない可能性があります。
* **解決策**: “Upload Context CSV” の列名 (例: Term、Definition、locale code) については、Locadex/GT のドキュメントまたは UI 内のヘルプを確認してください。CSV の列名を変更し、再度 upload してください。

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### 問題: 英語のままにすべき用語が翻訳されてしまう
</div>

* **原因**: 用語が Glossary に登録されていない、または「翻訳しない」が設定されていません (ロケール訳の欠落または誤り) 。
* **解決策**: 対象ロケールでも同じ値になるように、その用語を Glossary に追加します (例: “Artifacts” → ko: “Artifacts”) 。モデルがそれをプロダクト名または機能名として理解できるよう、短い Definition も追加します。

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### 問題: 日本語 (または別のロケール) では異なるルールが必要です
</div>

* **原因**: ロケール固有の慣例 (例: 丁寧語、スペーシング、プロダクト名のカタカナ表記) 。
* **解決策**: そのロケール用に別の Locale Context (例: ja) を追加し、必要に応じて「ja」列を含む追加の Glossary エントリ、または日本語向けの手動エントリを追加します。

<div id="cleanup-instructions">
  ## クリーンアップ手順
</div>

* コンソールのみで設定する場合、docs リポジトリに一時的なブランチやファイルは必要ありません。
* CSV を生成するための一時的なスクリプトを作成した場合、チームが保持すると判断しない限り、コミットしないでください (AGENTS.md と、一時的なスクリプトに関するユーザールールを参照してください) 。

<div id="checklist">
  ## チェックリスト
</div>

* [ ] human&#95;prompt、language&#95;dicts/ko.yaml (および該当する場合は ja) から用語を収集しました。
* [ ] Glossary CSV を作成または取得し、upload 用の列名を確認しました。
* [ ] Locadex コンソールにログインし、正しいプロジェクトを Open しました。
* [ ] Glossary 用語 (翻訳しない用語と翻訳済み用語) を upload または追加しました。
* [ ] 韓国語用の Locale Context を設定しました (該当する場合は後で日本語も設定) 。
* [ ] Style Controls (説明、対象読者、トーン) を設定しました。
* [ ] サンプル翻訳で検証し、必要に応じて既存のコンテンツを再翻訳しました。

<div id="glossary-csv">
  ## Glossary CSV
</div>

このリポジトリには、韓国語用のスターター Glossary が用意されています: **runbooks/locadex-glossary-ko.csv**。列:

* **Term**: ドキュメント内で表示されるソース (英語) の用語です。
* **Definition**: 短い説明です (AI に役立ちます。upload では optional) 。
* **ko**: 韓国語訳です。「翻訳しない」場合は Term と同じ文字列を使用し、「translate as」の場合は希望する韓国語の文字列を使用します。

`configs/language_dicts/ko.yaml` (または main 上の手動 KO ページ) からさらに用語を追加するには、同じ列で行を追記してください。Locadex コンソールがロケール翻訳に対して別の列名 (例: 「Translation (ko)」) を想定している場合は、upload 時または upload 前の CSV 内で `ko` 列の名前を変更してください。

<div id="csv-formatting-for-future-generation">
  ### 今後の生成に向けた CSV の書式
</div>

Glossary の CSV を作成または追記する際は (手動でもスクリプトでも) 、ファイルを有効な状態に保つため、次のルールに従ってください。

* **区切り文字**: コンマ (`,`) を使用します。フィールドが引用符で囲まれていない限り、フィールド内ではコンマを使用しないでください。
* **引用符**: フィールドにコンマ、二重引用符、または改行が含まれる場合は、そのフィールドを二重引用符 (`"`) で囲んでください。一貫性を保つために、すべてのフィールドを引用符で囲んでもかまいません。
* **エスケープ**: 引用符で囲まれたフィールド内で二重引用符そのものを表す場合は、二重引用符を 2 つ (`""`) 使用します。
* **1 行につき 1 用語**: 各行には 1 つの用語のみを記載します。1 つのセルに複数のバリエーションを記載しないでください (たとえば、Term 列には「`run`」と「`artifact`」をそれぞれ別の行に記載し、「`run, artifact`」のように 1 つのセルへまとめて記載しないでください) 。
* **ツール**: プログラムで CSV を生成する場合は、適切な CSV ライブラリ (たとえば Python の `csv` モジュールで `quoting=csv.QUOTE_MINIMAL` または `QUOTE_NONNUMERIC` を指定) を使用し、Term や Definition 内のコンマや引用符が正しく処理されるようにしてください。

<div id="notes">
  ## メモ
</div>

* **日本語対応は後で**: 日本語を追加する際は、`ja` 向けに Locale Context を改めて設定し (例: 丁寧語、アルファベットと日本語の間のスペース、インライン書式のスペース) 、`ja` 向けの Glossary エントリも追加してください (同じ考え方で、do-not-translate = ソースと同じ、translate-as = 意図する日本語) 。
* **Git の GT 設定**: `gt.config.json` には、すでに `locales` と `defaultLocale` があります。Glossary や AI context はそこには保存されず、コンソールにのみ保存されます。
* **参考資料**: [GT Glossary](https://generaltranslation.com/docs/platform/ai-context/glossary), [Locale Context](https://generaltranslation.com/docs/platform/ai-context/locale-context), [Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls), [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).