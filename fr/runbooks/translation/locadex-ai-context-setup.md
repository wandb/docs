<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # Prompt de l’agent : Configurer l’AI Context de Locadex pour la documentation de W&amp;B (coréen, puis japonais)
</div>

<div id="requirements">
  ## Exigences
</div>

* [ ] Accès au [General Translation Dashboard](https://dash.generaltranslation.com/) (console Locadex).
* [ ] Dépôt de documentation lié à un projet Locadex/GT (application GitHub installée, dépôt connecté).
* [ ] Facultatif : accès à la branche `main` de wandb/docs avec `ko/` (et éventuellement `ja/`) déjà présent, afin de comparer les traductions manuelles lors de l’affinage du glossaire ou du contexte local.

<div id="agent-prerequisites">
  ## Prérequis de l’agent
</div>

1. **Quelle(s) langue(s) configurez-vous ?** (par ex. coréen uniquement pour l’instant ; japonais plus tard.) Cela détermine quelles traductions du Glossaire et quelles entrées de Contexte local ajouter.
2. **Avez-vous déjà un CSV de Glossaire ou une liste de termes ?** Sinon, utilisez le runbook pour en créer un à partir des sources ci-dessous.
3. **Le projet GT est-il déjà créé et le dépôt connecté ?** Si ce n’est pas le cas, effectuez d’abord les [étapes 1–6 de Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).

<div id="task-overview">
  ## Aperçu de la tâche
</div>

Ce runbook explique comment extraire la mémoire de traduction et la terminologie à partir de (1) l’ancien outil `wandb_docs_translation` et (2) du contenu coréen (puis japonais) traduit manuellement sur `main`, ainsi que comment configurer la plateforme Locadex/General Translation afin que la traduction automatique s’appuie sur ce contexte. L’objectif est d’assurer une terminologie cohérente et un comportement « ne pas traduire » correct pour les noms de produits et les termes techniques.

**Où se trouvent les éléments :**

| Ce qui                                                      | Où                                             | Notes                                                                                                                                                         |
| ----------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Glossaire** (termes, définitions, traductions par locale) | Console Locadex → AI Context → Glossaire       | Assure un usage cohérent des termes et le comportement « ne pas traduire » pour les noms de produits et de fonctionnalités. Import en masse possible via CSV. |
| **Contexte local** (instructions spécifiques à la langue)   | Console Locadex → AI Context → Contexte local | p. ex. coréen : espacement entre alphabet latin et hangeul, règles de mise en forme.                                                                          |
| **Style Controls** (ton, audience, description du projet)   | Console Locadex → AI Context → Style Controls  | À l’échelle du projet ; s’applique à toutes les locales.                                                                                                      |
| **Fichiers/locales à traduire**                             | Git → `gt.config.json`                         | `locales`, `defaultLocale`, `files`. Aucun glossaire ni prompt dans le dépôt.                                                                                 |

Donc : **orientez la traduction automatique dans la console Locadex** (Glossaire, Contexte local, Style Controls). **La configuration des fichiers et des locales reste dans Git** (`gt.config.json`). La clé facultative `dictionary` dans `gt.config.json` est destinée aux chaînes de l’interface utilisateur de l’application (par ex. gt-next/gt-react), pas au glossaire MDX de la documentation ; la terminologie de la documentation est gérée dans la console.

<div id="context-and-constraints">
  ## Contexte et contraintes
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### Outils hérités (wandb_docs_translation)
</div>

* **human&#95;prompt.txt** : répertorie les noms de produits/fonctionnalités W&amp;B qui ne doivent **jamais** être traduits (à conserver en anglais) : Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models. Même principe dans les contextes de lien/liste, comme `[**word**](link)`.
* **system&#95;prompt.txt** : règles générales (markdown valide, traduire uniquement les commentaires dans les blocs de code, utiliser le dictionnaire, ne pas traduire les URL des liens ; pour le japonais/coréen : ajouter un espace lors du passage entre alphabets et caractères CJK, ainsi qu’autour du formatage en ligne).
* **configs/language&#95;dicts/ko.yaml** : « mémoire de traduction » mixte :
  * **Conserver en anglais** (nom de produit/fonctionnalité) : p. ex. `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`.
  * **Traduire en coréen** : p. ex. `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자.

La convention était donc la suivante : **les noms de produits/fonctionnalités (souvent avec une majuscule ou dans un contexte d’interface utilisateur/liste) restent en anglais** ; **les emplois comme noms communs** suivent le dictionnaire de la langue. Le Glossaire Locadex doit refléter à la fois « ne pas traduire » et « traduire par X » pour chaque langue.

<div id="locadexgt-platform-behavior">
  ### Comportement de la plateforme Locadex/GT
</div>

* **Glossaire** : terme (tel qu’il apparaît dans la source) + définition facultative + traduction facultative pour chaque locale. Pour « ne pas traduire », utilisez la même chaîne que le terme pour cette locale (par ex. terme « W&amp;B », traduction (ko) « W&amp;B »). Pour « traduire par », définissez Translation (ko) sur la traduction cible souhaitée (par ex. « artifact » → « 아티팩트 »).
* **Contexte local** : instructions en texte libre pour chaque locale cible (par ex. « Insérez un espace entre les caractères latins et coréens »).
* **Style Controls** : un seul ensemble de paramètres pour le projet (ton, audience, description). Appliqué à toutes les locales.
* Les modifications apportées à AI Context ne retraduisent **pas** automatiquement le contenu existant ; utilisez [Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate) pour appliquer le nouveau contexte aux fichiers déjà traduits.

<div id="step-by-step-process">
  ## Procédure étape par étape
</div>

<div id="1-gather-terminology-sources">
  ### 1. Recueillir les sources terminologiques
</div>

* **Dans wandb&#95;docs&#95;translation** (si disponible) :
  * `configs/human_prompt.txt` → liste des termes à ne jamais traduire.
  * `configs/language_dicts/ko.yaml` (puis `ja.yaml`) → table de correspondance des termes avec leur traduction pour la langue cible.
* **À partir des traductions manuelles sur main** (facultatif) : comparez quelques pages EN et KO (ou JA) pour vérifier comment les noms de produits et les termes courants ont été rendus (par ex. « run » vs « 실행 », « workspace » vs « 워크스페이스 »), puis ajoutez ou ajustez les entrées du Glossaire.

**Remarque pour l’agent** : si l’agent ne peut pas lire le dépôt externe, le runbook peut tout de même être suivi par une personne à l’aide du CSV et du texte de contexte de langue fournis dans ce dépôt (voir les runbooks et le CSV facultatif ci-dessous).

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. Créer ou obtenir un fichier CSV de glossaire
</div>

* Utilisez le fichier CSV de glossaire préconfiguré pour le coréen dans ce dépôt : **runbooks/locadex-glossary-ko.csv** (voir « Glossary CSV » ci-dessous), ou générez-en un qui inclut :
  * **Termes à ne pas traduire** : une ligne par terme ; définition facultative ; `ko` (ou « Translation (ko) ») = identique au terme.
  * **Termes traduits** : une ligne par terme ; définition facultative ; `ko` = équivalent coréen souhaité.
* Vérifiez les noms exacts des colonnes attendus par l’option « Upload Context CSV » de Locadex (par exemple `Term`, `Definition`, `ko` ou `Translation (ko)`). Ajustez les en-têtes du CSV si la console attend des noms différents.
* **Format CSV (pour un parsing valide)** : utilisez les règles standard de mise entre guillemets du format CSV afin que le fichier soit correctement analysé. La virgule est le séparateur de champs ; tout champ qui contient une virgule, un guillemet double ou un saut de ligne **doit** être entouré de guillemets doubles. Dans un champ entre guillemets, échappez les guillemets doubles internes en les doublant (`""`). Un terme par ligne (ne mettez pas plusieurs variantes comme « run, Run » dans une seule cellule). Lorsque vous générez ou modifiez le CSV par programmation, utilisez une bibliothèque CSV ou entourez explicitement ces champs de guillemets ; les virgules non protégées dans `Term` ou `Definition` seront traitées comme des séparateurs de colonnes et rendront la ligne invalide.

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. Configurez le projet Locadex dans la console
</div>

1. Connectez-vous au [General Translation Dashboard](https://dash.generaltranslation.com/).
2. Ouvrez le projet lié au dépôt wandb/docs.
3. Accédez à **AI Context** (ou à l’équivalent : Glossaire, Contexte local, Style Controls).

<div id="4-upload-or-add-glossary-terms">
  ### 4. Téléverser ou ajouter des termes du glossaire
</div>

* **Option A** : utilisez **Upload Context CSV** pour importer le glossaire en masse (Term, Definition et la ou les colonnes de langue). La plateforme associe les colonnes aux termes du glossaire et aux traductions pour chaque langue.
* **Option B** : ajoutez les termes manuellement : Term, Definition (pour aider le modèle) et, pour le coréen, ajoutez la traduction (identique au terme pour « ne pas traduire », ou la chaîne coréenne pour « traduire par »).

Assurez-vous d’avoir au minimum :

* Les noms de produits/fonctionnalités qui doivent rester en anglais : W&amp;B, Weights &amp; Biases, Artifacts, Runs, Experiments, Sweeps, Weave, Launch, Models, Reports, Datasets, Teams, Users, Workspace, Registered Models, etc., avec Korean = identique à la source.
* Les termes qui doivent être traduits de manière cohérente : p. ex. artifact → 아티팩트, sweep → 스윕, project → 프로젝트, workspace → 워크스페이스, ainsi que les autres entrées de `language_dicts/ko.yaml` (et plus tard `ja.yaml`).

<div id="5-set-locale-context-for-korean">
  ### 5. Définir le Contexte local pour le coréen
</div>

* Sélectionnez la locale **ko**.
* Ajoutez des instructions qui reflètent l’ancien `system_prompt` et les bonnes pratiques pour la documentation en coréen, par exemple :
  * Ajoutez une espace lors du passage entre des lettres latines et des caractères coréens (y compris le hangul et les hanja).
  * Lorsque vous utilisez une mise en forme en ligne (gras, italique, code) autour d’une partie d’un mot ou d’une expression en coréen, ajoutez des espaces avant et après la partie mise en forme afin que le Markdown soit correctement rendu.
  * Conservez les blocs de code et les URL de lien inchangés ; traduisez uniquement le texte environnant et, le cas échéant, les commentaires dans le code.

Enregistrez le Contexte local.

<div id="6-set-style-controls-project-wide">
  ### 6. Définir les Style Controls (pour l’ensemble du projet)
</div>

* **Description du projet** : par ex. « Documentation de Weights &amp; Biases (W&amp;B) : suivi des expériences de ML, registre de modèles, Weave pour les opérations LLM et les produits associés. »
* **Public cible** : développeurs et praticiens du ML.
* **Ton** : professionnel, technique, clair. Privilégiez une formulation naturelle plutôt qu’une traduction littérale.

Enregistrez.

<div id="7-retranslate-if-needed">
  ### 7. Retraduisez si nécessaire
</div>

* Si vous disposez déjà de contenu traduit automatiquement et que vous avez modifié le Glossaire ou le Contexte local, utilisez le processus **Retranslate** de la plateforme pour les fichiers concernés afin d’appliquer le nouveau contexte.

<div id="verification-and-testing">
  ## Vérification et tests
</div>

* **Glossaire** : après le téléversement, vérifiez quelques termes dans l’onglet Glossaire (do-not-translate et translated).
* **Contexte local** : confirmez que les instructions en coréen (et plus tard en japonais) sont enregistrées pour la bonne locale.
* **Qualité** : lancez ou déclenchez la traduction sur une page d’exemple et vérifiez que les noms de produits restent en anglais et que les termes courants correspondent au glossaire (par ex. artefact → 아티팩트, le cas échéant).

<div id="common-issues-and-solutions">
  ## Problèmes courants et solutions
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### Problème : le téléversement du CSV n’est pas associé au Glossaire
</div>

* **Cause** : Les noms de colonnes peuvent ne pas correspondre à ceux attendus par la plateforme.
* **Solution** : Consultez la documentation Locadex/GT ou l’aide intégrée à l’interface utilisateur pour connaître les noms de colonnes de « Upload Context CSV » (par exemple, Term, Definition, code de langue). Renommez les colonnes dans votre CSV, puis téléversez-le à nouveau.

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### Problème : certains termes sont toujours traduits alors qu’ils devraient rester en anglais
</div>

* **Cause** : le terme ne figure pas dans le Glossaire, ou l’option « do not translate » n’est pas définie (traduction de la langue cible manquante ou incorrecte).
* **Solution** : ajoutez le terme au Glossaire avec la même valeur pour la langue cible (par ex. « Artifacts » → ko : « Artifacts »). Ajoutez une courte Definition afin que le modèle comprenne qu’il s’agit d’un nom de produit ou de fonctionnalité.

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### Problème : le japonais (ou une autre locale) nécessite des règles différentes
</div>

* **Cause** : préférences spécifiques à la locale (par ex. forme polie, espacement, katakana pour les noms de produits).
* **Solution** : ajoutez un Contexte local distinct pour cette locale (par ex. `ja`) et, si nécessaire, des entrées supplémentaires du Glossaire avec une colonne « ja » ou des entrées manuelles pour le japonais.

<div id="cleanup-instructions">
  ## Consignes de nettoyage
</div>

* Aucune branche ni aucun fichier temporaire ne sont nécessaires dans le dépôt de documentation pour une configuration effectuée uniquement depuis la console.
* Si vous avez généré un script ad hoc pour créer le CSV, ne l’incluez pas dans un commit, sauf si l’équipe décide de le conserver (voir AGENTS.md et les règles utilisateur concernant les scripts ponctuels).

<div id="checklist">
  ## Liste de contrôle
</div>

* [ ] Terminologie recueillie à partir de `human_prompt`, de `language_dicts/ko.yaml` (et de `ja`, le cas échéant).
* [ ] Glossaire CSV créé ou obtenu, et noms de colonnes vérifiés pour le téléversement.
* [ ] Connexion à la console Locadex effectuée et bon projet ouvert.
* [ ] Termes du Glossaire téléversés ou ajoutés (à ne pas traduire et traduits).
* [ ] Contexte local défini pour le coréen (et ensuite pour le japonais, le cas échéant).
* [ ] Style Controls définis (description, audience, ton).
* [ ] Vérification effectuée à l’aide d’un exemple de traduction, et contenu existant retraduit si nécessaire.

<div id="glossary-csv">
  ## Glossaire CSV
</div>

Un glossaire coréen de départ est fourni dans ce dépôt : **runbooks/locadex-glossary-ko.csv**. Colonnes :

* **Term** : terme source (anglais) tel qu’il apparaît dans la documentation.
* **Definition** : brève explication (utile pour l’IA ; facultative lors de l’import).
* **ko** : traduction coréenne. Utilisez la même chaîne que Term pour « ne pas traduire » ; utilisez la chaîne coréenne souhaitée pour « traduire par ».

Pour ajouter d’autres termes depuis `configs/language_dicts/ko.yaml` (ou depuis des pages KO manuelles sur main), ajoutez des lignes avec les mêmes colonnes. Si la console Locadex attend des noms de colonnes différents pour les traductions de paramètres régionaux (par exemple « Translation (ko) »), renommez la colonne `ko` lors de l’import ou dans le CSV avant l’import.

<div id="csv-formatting-for-future-generation">
  ### Formatage CSV pour les générations futures
</div>

Lorsque vous créez le CSV du glossaire ou y ajoutez du contenu (manuellement ou par script), respectez les règles suivantes afin que le fichier reste valide :

* **Délimiteur** : virgule (`,`). N’utilisez pas de virgule dans un champ, sauf si ce champ est entre guillemets.
* **Guillemets** : mettez un champ entre guillemets doubles (`"`) s’il contient une virgule, un guillemet double ou un saut de ligne. Vous pouvez également mettre tous les champs entre guillemets pour plus de cohérence.
* **Échappement** : dans un champ entre guillemets, représentez un guillemet double littéral par deux guillemets doubles (`""`).
* **Un terme par ligne** : chaque ligne correspond à un terme. N’indiquez pas plusieurs variantes dans une même cellule (par exemple, utilisez des lignes distinctes pour « run » et « artifact », et non « run, artifact » dans la colonne Term).
* **Outils** : lorsque vous générez un CSV par programmation, utilisez une bibliothèque CSV appropriée (par exemple, le module Python `csv` avec `quoting=csv.QUOTE_MINIMAL` ou `QUOTE_NONNUMERIC`) afin que les virgules et les guillemets dans Term ou Definition soient correctement gérés.

<div id="notes">
  ## Notes
</div>

* **Japonais plus tard** : lors de l’ajout du japonais, répétez le Contexte local pour `ja` (par ex. forme polie, espaces entre l’alphabet latin et l’écriture japonaise, espaces dans le formatage en ligne) et ajoutez des entrées de Glossaire pour `ja` (même approche : do-not-translate = identique à la source ; translate-as = traduction japonaise souhaitée).
* **Configuration GT dans Git** : `gt.config.json` contient déjà `locales` et `defaultLocale`. Aucun glossaire ni AI Context n’y est stocké ; ils se trouvent uniquement dans la console.
* **Références** : [Glossaire GT](https://generaltranslation.com/docs/platform/ai-context/glossary), [Contexte local](https://generaltranslation.com/docs/platform/ai-context/locale-context), [Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls), [Locadex pour Mintlify](https://generaltranslation.com/docs/locadex/mintlify).