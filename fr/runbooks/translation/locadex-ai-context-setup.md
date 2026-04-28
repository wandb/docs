---
title: Configuration du contexte IA de Locadex
---

<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # Prompt de l’agent : Configurer le contexte Locadex AI pour la documentation W&amp;B (en coréen, puis en japonais)
</div>

<div id="requirements">
  ## Prérequis
</div>

* [ ] Accès au [Dashboard General Translation](https://dash.generaltranslation.com/) (console Locadex).
* [ ] Le dépôt de documentation lié à un projet Locadex/GT (application GitHub installée, dépôt connecté).
* [ ] Facultatif : accès à la branche `main` de wandb/docs, avec `ko/` (et éventuellement `ja/`), afin de comparer les traductions manuelles lors de l’affinage du Glossaire ou du Contexte local.

<div id="agent-prerequisites">
  ## Prérequis de l’agent
</div>

1. **Quelle(s) langue(s) paramétrez-vous ?** (par ex. coréen uniquement pour l’instant ; japonais plus tard.) Cela détermine quelles traductions du Glossaire et quelles entrées du Contexte local ajouter.
2. **Avez-vous déjà un fichier CSV du Glossaire ou une liste de termes ?** Sinon, utilisez le runbook pour en créer un à partir des sources ci-dessous.
3. **Le projet GT est-il déjà créé et le dépôt connecté ?** Sinon, terminez d’abord les étapes 1 à 6 de [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).

<div id="task-overview">
  ## Aperçu de la tâche
</div>

Ce runbook explique comment récupérer la mémoire de traduction et la terminologie depuis (1) l’ancien outil `wandb_docs_translation` et (2) le contenu coréen traduit manuellement (puis, plus tard, le contenu japonais) sur `main`, ainsi que comment configurer la plateforme Locadex/General Translation pour que la traduction automatique s’appuie sur ce contexte. L’objectif est d’assurer une terminologie cohérente et un comportement correct de type « ne pas traduire » pour les noms de produits et les termes techniques.

**Où se trouvent les éléments :**

| Quoi                                                          | Où                                                           | Notes                                                                                                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Glossaire** (termes, définitions, traductions par langue)   | Console Locadex → AI Context → Glossary                      | Garantit un usage cohérent des termes et le comportement « ne pas traduire » pour les noms de produits et de fonctionnalités. Import en masse possible via CSV. |
| **Contexte local** (instructions propres à la langue)         | Console Locadex → AI Context → Locale Context                | p. ex. pour le coréen : espacement entre alphabet latin et hangul, règles de formatage.                                                                         |
| **Contrôle de style** (ton, audience, description du projet) | Console Locadex → AI Context → Style Controls                | À l’échelle du projet ; s’applique à toutes les langues.                                                                                                        |
| **Quels fichiers/langues traduire**                           | Git → `gt.config.json`                                       | `locales`, `defaultLocale`, `files`. Aucun glossaire ni prompt dans le dépôt.                                                                                   |
| **Journal des problèmes fournisseur (bugs Locadex)**          | Git → [locadex-vendor-issues.md](./locadex-vendor-issues.md) | Suivez les défauts de traduction à signaler à Locadex (par exemple, des URL altérées dans des fichiers MDX localisés).                                          |

Donc : **pilotez la traduction automatique depuis la console Locadex** (Glossaire, Contexte local, Contrôle de style). **La configuration des fichiers et des langues reste dans Git** (`gt.config.json`). La clé `dictionary` facultative dans `gt.config.json` est destinée aux chaînes de l’interface utilisateur de l’application (par ex. gt-next/gt-react), et non au glossaire MDX de la documentation ; la terminologie de la documentation est gérée dans la console.

<div id="context-and-constraints">
  ## Contexte et contraintes
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### Outils existants (wandb_docs_translation)
</div>

* **human&#95;prompt.txt** : répertorie les noms de produits/fonctionnalités W&amp;B qui ne doivent **jamais** être traduits (à laisser en anglais) : Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models. Même règle dans les contextes de lien/liste comme `[**word**](link)`.
* **system&#95;prompt.txt** : règles générales (markdown valide, traduire uniquement les commentaires dans les blocs de code, utiliser le dictionnaire, ne pas traduire les URL des liens ; pour le japonais/coréen : ajouter une espace lors du passage entre alphabets et caractères CJK, ainsi qu&#39;autour de la mise en forme en ligne).
* **configs/language&#95;dicts/ko.yaml** : « mémoire de traduction » mixte :
  * **À conserver en anglais** (nom de produit/fonctionnalité) : p. ex. `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`.
  * **À traduire en coréen** : p. ex. `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자.

La convention était donc la suivante : **les noms de produits/fonctionnalités (souvent avec une majuscule ou dans un contexte d&#39;UI/liste) restent en anglais** ; **les emplois comme noms communs** suivent le dictionnaire de la langue. Le glossaire Locadex doit refléter à la fois « ne pas traduire » et « traduire par X » pour chaque langue.

<div id="locadexgt-platform-behavior">
  ### Comportement de la plateforme Locadex/GT
</div>

* **Glossaire** : Terme (tel qu’il apparaît dans la source) + Définition facultative + Traduction facultative par langue. Pour « ne pas traduire », utilisez la même chaîne que le terme pour cette langue (par ex. Term « W&amp;B », Translation (ko) « W&amp;B »). Pour « traduire par », définissez Translation (ko) sur la valeur cible souhaitée (par ex. « artifact » → « 아티팩트 »).
* **Contexte local** : Instructions libres pour chaque langue cible (par ex. « Utiliser une espace entre les caractères latins et coréens »).
* **Contrôles de style** : Un seul ensemble pour le projet (ton, audience, description). Il s’applique à toutes les langues.
* Les modifications du contexte IA **ne** retraduisent **pas** automatiquement le contenu existant ; utilisez [Retraduire](https://generaltranslation.com/docs/platform/translations/retranslate) pour appliquer le nouveau contexte aux fichiers déjà traduits.

<div id="step-by-step-process">
  ## Processus étape par étape
</div>

<div id="1-gather-terminology-sources">
  ### 1. Rassembler les sources terminologiques
</div>

* **À partir de wandb&#95;docs&#95;translation** (si disponible) :
  * `configs/human_prompt.txt` → liste des termes à ne jamais traduire.
  * `configs/language_dicts/ko.yaml` (puis `ja.yaml`) → correspondance terme → traduction pour la locale cible.
* **À partir des traductions manuelles sur main** (facultatif) : comparez quelques pages EN et KO (ou JA) pour vérifier comment les noms de produit et les termes courants ont été traduits (par ex. « run » vs « 실행 », « Workspace » vs « 워크스페이스 »), puis ajoutez ou ajustez les entrées du glossaire.

**Note de l’agent** : si l’agent ne peut pas lire le dépôt externe, un humain peut quand même suivre cette procédure à l’aide du CSV et du texte de contexte de locale fournis dans ce dépôt (voir les runbooks et le CSV facultatif ci-dessous).

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. Créer ou obtenir un CSV de glossaire
</div>

* Utilisez le CSV de glossaire préconfiguré pour le coréen dans ce dépôt : **runbooks/locadex-glossary-ko.csv** (voir « CSV de glossaire » ci-dessous), ou générez-en un qui inclut :
  * **Termes à ne pas traduire** : une ligne par terme ; définition facultative ; `ko` (ou « Translation (ko) ») = identique au terme.
  * **Termes traduits** : une ligne par terme ; définition facultative ; `ko` = équivalent coréen souhaité.
* Vérifiez les noms de colonnes exacts attendus par « Upload Context CSV » dans Locadex (par ex. `Term`, `Definition`, `ko` ou `Translation (ko)`). Ajustez les en-têtes du CSV si la console attend des noms différents.
* **Format CSV (pour un parsing correct)** : utilisez les règles standard de mise entre guillemets du CSV afin que le fichier soit correctement interprété. La virgule est le séparateur de champs ; tout champ contenant une virgule, un guillemet double ou un saut de ligne **doit** être entouré de guillemets doubles. Dans un champ entre guillemets, échappez les guillemets doubles internes en les doublant (`""`). Un terme par ligne (ne mettez pas plusieurs variantes comme « run, Run » dans une seule cellule). Lorsque vous générez ou modifiez le CSV par programmation, utilisez une bibliothèque CSV ou mettez explicitement ces champs entre guillemets ; les virgules non protégées par des guillemets dans `Term` ou `Definition` seront traitées comme des séparateurs de colonnes et rendront la ligne invalide.

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. Configurez le projet Locadex dans la console
</div>

1. Connectez-vous au [General Translation Dashboard](https://dash.generaltranslation.com/).
2. Ouvrez le projet associé au dépôt wandb/docs.
3. Accédez à **AI Context** (ou à l’équivalent : Glossaire, Contexte local, Contrôles de style).

<div id="4-upload-or-add-glossary-terms">
  ### 4. Importer ou ajouter des termes du Glossaire
</div>

* **Option A** : Utilisez **Upload Context CSV** pour importer le glossaire en masse (Term, Definition et la ou les colonnes de langue). La plateforme associe les colonnes aux termes du glossaire et aux traductions propres à chaque langue.
* **Option B** : Ajoutez les termes manuellement : Term, Definition (pour aider le modèle) et, pour le coréen, ajoutez la traduction (identique au terme pour « do not translate », ou la chaîne coréenne pour « translate as »).

Assurez-vous d’avoir au minimum :

* Les noms de produits/fonctionnalités qui doivent rester en anglais : W&amp;B, Weights &amp; Biases, Artifacts, Runs, Experiments, Sweeps, Weave, Launch, Models, Reports, Datasets, Teams, Users, Workspace, Registered Models, etc., avec Korean = identique à la source.
* Les termes qui doivent être traduits de manière cohérente : p. ex. artifact → 아티팩트, sweep → 스윕, project → 프로젝트, workspace → 워크스페이스, ainsi que les autres entrées de `language_dicts/ko.yaml` (et plus tard `ja.yaml`).

<div id="5-set-locale-context-for-korean">
  ### 5. Définir le Contexte local pour le coréen
</div>

* Sélectionnez la langue **ko**.
* Ajoutez des instructions reflétant le `system_prompt` historique et les bonnes pratiques de la documentation en coréen, par exemple :
  * Ajoutez une espace lors du passage entre des lettres latines et des caractères coréens (y compris le hangeul et les hanja).
  * Lorsque vous utilisez une mise en forme en ligne (gras, italique, code) sur une partie d’un mot ou d’une expression en coréen, ajoutez des espaces avant et après la partie mise en forme afin que le markdown soit correctement rendu.
  * Conservez les blocs de code et les URL des liens inchangés ; traduisez uniquement le texte alentour et, le cas échéant, les commentaires dans le code.

Enregistrez le Contexte local.

<div id="6-set-style-controls-project-wide">
  ### 6. Définir les paramètres de style (à l’échelle du projet)
</div>

* **Description du projet** : p. ex. « Documentation de Weights &amp; Biases (W&amp;B) : suivi des expériences ML, registre de modèles, Weave pour les ops LLM et produits associés. »
* **Public cible** : développeurs et praticiens du ML.
* **Ton** : professionnel, technique, clair. Privilégiez une formulation naturelle plutôt qu’une traduction littérale.

Enregistrer.

<div id="7-retranslate-if-needed">
  ### 7. Retraduisez si nécessaire
</div>

* Si vous avez déjà du contenu traduit automatiquement et que vous avez modifié le Glossaire ou le Contexte local, utilisez l’option **Retraduire** de la plateforme pour les fichiers concernés afin que le nouveau contexte soit pris en compte.

<div id="verification-and-testing">
  ## Vérification et tests
</div>

* **Glossaire** : Après l’import, vérifiez quelques termes au hasard dans l’onglet Glossaire (ceux à ne pas traduire et ceux traduits).
* **Contexte local** : Vérifiez que les instructions en coréen (et plus tard en japonais) sont bien enregistrées sous la bonne locale.
* **Qualité** : Exécutez ou déclenchez la traduction sur un exemple de page, puis vérifiez que les noms de produit restent en anglais et que les termes courants correspondent au glossaire (par exemple, artifact → 아티팩트 lorsque c’est approprié).

<div id="common-issues-and-solutions">
  ## Problèmes fréquents et solutions
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### Problème : l’upload du CSV ne correspond pas au Glossaire
</div>

* **Cause** : les noms de colonnes peuvent ne pas correspondre à ce que la plateforme attend.
* **Solution** : consultez la documentation Locadex/GT ou l’aide dans l’UI pour connaître les noms de colonnes attendus pour “Upload Context CSV” (par ex. Term, Definition, locale code). Renommez les colonnes de votre CSV, puis relancez l’upload.

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### Problème : certains termes sont encore traduits alors qu’ils devraient rester en anglais
</div>

* **Cause** : le terme ne figure pas dans le Glossaire, ou l’option « ne pas traduire » n’est pas définie (traduction pour la langue absente ou incorrecte).
* **Solution** : ajoutez le terme au Glossaire avec la même valeur pour la langue cible (par ex. « Artifacts » → ko: « Artifacts »). Ajoutez une brève définition afin que le modèle comprenne qu’il s’agit d’un nom de produit ou de fonctionnalité.

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### Problème : le japonais (ou une autre langue) nécessite des règles différentes
</div>

* **Cause** : des préférences propres à la langue concernée (par ex. niveau de politesse, espacement, katakana pour les noms de produits).
* **Solution** : ajoutez un Contexte local distinct pour cette langue (par ex. `ja`) et, si nécessaire, des entrées supplémentaires dans le Glossaire avec une colonne « ja » ou des entrées manuelles pour le japonais.

<div id="cleanup-instructions">
  ## Consignes de nettoyage
</div>

* Aucune branche ni aucun fichier temporaires ne sont nécessaires dans le dépôt de documentation pour une configuration effectuée uniquement dans la console.
* Si vous avez généré un script ponctuel pour créer le CSV, ne le validez pas, sauf si l’équipe décide de le conserver (voir AGENTS.md et les règles utilisateur concernant les scripts ponctuels).

<div id="checklist">
  ## Liste de contrôle
</div>

* [ ] Terminologie recueillie à partir de `human_prompt`, `language_dicts/ko.yaml` (et `ja`, le cas échéant).
* [ ] Fichier CSV du Glossaire créé ou obtenu, et noms de colonnes confirmés pour l’upload.
* [ ] Connexion à la console Locadex et ouverture du bon projet.
* [ ] Upload ou ajout des termes du Glossaire (à ne pas traduire et traduits).
* [ ] Définition du Contexte local pour le coréen (et, plus tard, le japonais, le cas échéant).
* [ ] Définition des contrôles de style (description, audience, ton).
* [ ] Vérification à l’aide d’un exemple de traduction et retraduction du contenu existant si nécessaire.

<div id="glossary-csv">
  ## CSV du glossaire
</div>

Un glossaire coréen de base est fourni dans ce dépôt : **runbooks/locadex-glossary-ko.csv**. Colonnes :

* **Term** : terme source (anglais) tel qu’il apparaît dans la documentation.
* **Definition** : brève explication (utile pour l’IA ; facultatif pour l’import).
* **ko** : traduction en coréen. Utilisez la même chaîne que Term pour « ne pas traduire » ; utilisez la chaîne coréenne souhaitée pour « traduire par ».

Pour ajouter d’autres termes à partir de `configs/language_dicts/ko.yaml` (ou de pages KO manuelles sur main), ajoutez des lignes avec les mêmes colonnes. Si la console Locadex attend des noms de colonnes différents pour les traductions de langue (par exemple « Translation (ko) »), renommez la colonne `ko` lors de l’import ou dans le CSV avant l’import.

<div id="csv-formatting-for-future-generation">
  ### Formatage CSV pour une génération ultérieure
</div>

Lorsque vous créez le fichier CSV du glossaire ou y ajoutez du contenu (manuellement ou par script), respectez les règles suivantes afin que le fichier reste valide :

* **Délimiteur** : virgule (`,`). N’utilisez pas de virgule dans un champ, sauf si ce champ est entre guillemets.
* **Guillemets** : Mettez un champ entre guillemets doubles (`"`) s’il contient une virgule, un guillemet double ou un saut de ligne. Vous pouvez également mettre tous les champs entre guillemets par souci de cohérence.
* **Échappement** : Dans un champ entre guillemets, représentez un guillemet double littéral par deux guillemets doubles (`""`).
* **Un terme par ligne** : chaque ligne correspond à un terme. N’indiquez pas plusieurs variantes dans une même cellule (par ex., utilisez des lignes distinctes pour « run » et « artifact », et non « run, artifact » dans la colonne Term).
* **Outils** : lorsque vous générez un CSV par programmation, utilisez une bibliothèque CSV adaptée (par ex. le module Python `csv` avec `quoting=csv.QUOTE_MINIMAL` ou `QUOTE_NONNUMERIC`) afin que les virgules et les guillemets dans Term ou Definition soient correctement gérés.

<div id="notes">
  ## Notes
</div>

* **Japonais ultérieurement** : lors de l’ajout du japonais, répétez le Contexte local pour `ja` (par exemple : forme polie, espacement entre l’alphabet latin et l’écriture japonaise, espaces pour la mise en forme en ligne) et ajoutez des entrées de Glossaire pour `ja` (même approche : ne pas traduire = identique à la source ; traduire par = japonais souhaité).
* **configuration GT dans Git** : `gt.config.json` contient déjà `locales` et `defaultLocale`. Aucun glossaire ni contexte IA n’y est stocké ; ils existent uniquement dans la console.
* **Références** : [GT Glossaire](https://generaltranslation.com/docs/platform/ai-context/glossary), [Contexte local](https://generaltranslation.com/docs/platform/ai-context/locale-context), [Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls), [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).