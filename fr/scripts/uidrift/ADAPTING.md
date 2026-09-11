---
title: ADAPTATION
---

<div id="adapting-this-detector-to-another-repo">
  # Adapter ce détecteur à un autre dépôt
</div>

Ceci est un exemple concret, pas un framework. Il n&#39;y a aucune interface de plugin à
implémenter ni de classe de base abstraite à dériver — il faudrait pour cela deviner
la forme de la deuxième implémentation avant que quiconque en ait écrit une. Ce
fichier indique plutôt ce qui est générique, ce qui est spécifique à `wandb/core` et ce qui nous a
surpris, afin que l&#39;adapter relève de la lecture plutôt que de l&#39;archéologie.

Si vous êtes un agent à qui l&#39;on demande *« fais ceci, mais surveille les versions de
`coreweave/sunk` »* — lisez d&#39;abord ce fichier, puis `config.py`, puis `extract.py`. Les
autres modules en découlent.

<div id="the-four-part-anatomy">
  ## L&#39;anatomie en quatre parties
</div>

Tout détecteur de dérive documentaire surveillant un dépôt comporte les mêmes quatre parties. Seule la deuxième
colonne change.

| Partie                            | Générique — réutilisable tel quel                                                       | Spécifique au dépôt — à réécrire                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **1. Détecteur d&#39;événements** | itération sur les commits, découpage du diff, déduplication par égalité d&#39;ensembles | quels chemins sont exposés aux utilisateurs ; quels motifs littéraux servent de libellés |
| **2. Preuves**                    | signaux structurels, portée des gates, propriété                                    | emplacement et mécanisme du registre des gates ; structure du fichier CODEOWNERS     |
| **3. Triage**                     | la procédure de décision agent/binôme/humain                                            | seuils ; quels chemins sont immuables                                                    |
| **4. Destination**                | moteur de rendu de tableaux markdown, registre, PR de rapport glissant                   | emplacement du rapport ; projet et composant JIRA                                        |

C&#39;est dans la partie 1 que se concentre la quasi-totalité du coût d&#39;adaptation. Les parties 3 et 4 se
transposent généralement sans modification.

<div id="step-zero-for-a-new-repo-is-there-an-i18n-catalog">
  ## Étape zéro pour un nouveau dépôt : existe-t-il un catalogue i18n ?
</div>

**Posez-vous cette question avant toute autre chose.** C&#39;est elle qui détermine si le travail prendra deux jours
ou deux semaines.

* **Un catalogue existe** (`en.json`, `messages.po`, `.ftl`, une configuration i18next/Lingui/
  react-intl) : vous comparez des paires clé-valeur structurées. Les clés vous donnent
  gratuitement une identité stable lors des renommages, et vous pouvez faire l&#39;impasse sur la majeure partie de `extract.py`.
  C&#39;est le cas facile.
* **Aucun catalogue** — le cas de `wandb/core` : les chaînes sont écrites directement dans le JSX et vous
  analysez le code source. Tout ce qui suit s&#39;applique.

Comment le vérifier rapidement :

```bash
git -C <repo> grep -lE 'useTranslation|defineMessages|FormattedMessage|i18next|@lingui' <ref> | head
git -C <repo> ls-tree -r --name-only <ref> | grep -iE 'locales?/|translations?/|/en\.json$'
```

Pour `wandb/core` sur `origin/master`, les deux ne renvoient rien. Le dépôt ne
contient aucune boîte à outils i18n. (Il existe bien un projet pilote de
localisation Locadex/gt-react, mais il s&#39;exécute sur un *fork* —
`wandb/mattcore` — et n&#39;affecte pas `wandb/core`.)

<div id="what-surprised-us-on-wandbcore">
  ## Ce qui nous a surpris sur wandb/core
</div>

Voici les constats qui nous ont fait perdre un temps précieux. C&#39;est pour cette raison que ce fichier existe.

<div id="1-enumerating-attribute-names-guarantees-silent-misses">
  ### 1. Énumérer les noms d&#39;attributs garantit des oublis silencieux
</div>

Le premier extracteur listait les attributs porteurs de texte : `aria-label`,
`placeholder`, `title`, `tooltip`. Il a obtenu un score de **zéro** sur un commit de
consolidation de volet latéral, car cette bibliothèque de composants reçoit son texte via `saveLabel=`,
`cancelLabel=`, `isPendingAriaLabel=`. Un design system invente de nouvelles props de libellé
à mesure qu&#39;il évolue, et la liste reste ouverte.

**Faites correspondre le suffixe, pas l&#39;appartenance à une liste.** Voir `_ATTR_SUFFIX` dans `extract.py`.
En contrepartie, `name=` devient ambigu (`<Icon name="info" />` est un
identifiant, `<Hotkey name="List only visible runs" />` est du texte) ; le problème est traité en
rejetant les valeurs en forme de slug pour les seules clés ambiguës.

<div id="2-prettier-reflow-is-the-dominant-false-positive">
  ### 2. Le reformatage de Prettier est le principal faux positif
</div>

Une réindentation apparaît sous la forme `-aria-label="X"` / `+  aria-label="X"` — une suppression
et un ajout de la même chaîne. Ce cas est éliminé de façon déterministe par une égalité
d&#39;ensembles fichier par fichier, sans modèle ni heuristique. Environ 25 des 233 commits touchant
des libellés sur une fenêtre de 60 jours sont de purs reformatages.

Généralisé à partir de `diff_signals.graphql_contract_change`, qui recourt à la même
astuce pour déterminer si une modification d&#39;un fichier `.graphql` est visible côté client.

<div id="3-refactor-titled-commits-are-the-dominant-false-negative">
  ### 3. Les commits intitulés « refactor » constituent le principal faux négatif
</div>

**Ne filtrez jamais sur le type de commit conventionnel.** Le constat le plus riche du corpus — huit en-têtes de tableau passés en casse de titre, toujours erronés dans la documentation publiée six semaines plus tard — provenait pour moitié de `feat(app): migrate ... to Table` et pour moitié de `refactor(app): migrate OrgDashboard UsersTable`. Aucune de ces lignes de sujet ne laisse supposer qu&#39;un texte visible par l&#39;utilisateur a été modifié. Les deux l&#39;ont pourtant fait.

Le type de commit est enregistré en tant que metadata, mais n&#39;est exploité par rien.

<div id="4-never-normalize-case">
  ### 4. Ne jamais normaliser la casse
</div>

`normalize()` réduit les espaces et s&#39;arrête là. Passer en minuscules rendrait
`MODELS SEAT` et `Models Seat` identiques : le renommage ne portant que sur la casse
s&#39;annulerait de lui-même dans l&#39;arithmétique des ensembles et disparaîtrait sans laisser de trace.
Un test verrouille ce comportement (`test_case_is_never_normalized`), justement parce que l&#39;échec est
silencieux.

<div id="5-move-detection-needs-a-looser-identity-than-reflow-detection">
  ### 5. La détection des déplacements exige une identité *plus souple* que la détection de reformatage
</div>

Ce sont deux questions distinctes, et elles appellent deux clés distinctes :

* *Le texte de ce fichier a-t-il changé ?* → identité stricte `(kind, key, string)`.
  Le reformatage préserve exactement la forme de l&#39;expression : la forme doit donc entrer en jeu.
* *Cette chaîne a-t-elle disparu du produit ?* → la chaîne seule.

Une consolidation de volet latéral a déplacé `Add secret` hors de `<span>Add secret</span>`
vers `saveLabel="Add secret"`. Même chaîne, forme différente, toujours affichée à l&#39;écran.
Avec une clé stricte, ce commit signale 23 suppressions fantômes — soit 23 lignes erronées dans
le tout premier rapport que l&#39;on consulte. Voir `LabelDelta.ident` et `.moved_ident`.

<div id="6-wrapped-means-not-a-complete-literal-not-prettier-moved-it">
  ### 6. `wrapped` signifie « littéral incomplet », et non « Prettier l&#39;a déplacé »
</div>

La confusion est facile à faire, et elle disqualifie sans raison de bons constats de la voie agent. L&#39;interpolation (`` `Allow ${AGENT_NAME} to ...` ``) et les branches de ternaires ne peuvent réellement pas faire l&#39;objet d&#39;un rechercher-remplacer. En revanche, le texte que Prettier a renvoyé sur sa propre ligne est capturé à l&#39;identique et ne présente aucun risque.

<div id="7-merged-visible-and-flag-presence-is-a-decayed-signal">
  ### 7. Fusionné ≠ visible, et la *présence* d&#39;un flag est un signal dégradé
</div>

Les nouveautés de l&#39;interface utilisateur sont livrées derrière des flags de déploiement progressif Statsig. Mais les ingénieurs suppriment rarement un flag une fois qu&#39;il a atteint 100 % — le laisser en place est plus sûr — si bien que la présence d&#39;un gate ne vous apprend presque rien. Ne l&#39;utilisez pas comme critère de filtrage.

Ce sont les **événements de cycle de vie** qui constituent le signal :

| Événement                                        | Signification            |
| ------------------------------------------------ | ------------------------ |
| Flag ajouté dans le même commit que le texte     | pas encore visible       |
| Commit de suppression du flag                    | GA                       |
| Flag simplement présent, ajouté il y a longtemps | aucun signal — à ignorer |

Le gating s&#39;applique par ailleurs **par surface, et non par feature** : un même gate régit trois surfaces dans `APIKeysTabContent.tsx`, avec une réponse différente pour chacune. Le diff indique si l&#39;élément modifié se trouve à l&#39;intérieur de la condition ; appuyez-vous sur cela, et non sur le nom du flag.

La sémantique de déploiement propre à `wandb/core` est suffisamment complexe pour faire l&#39;objet d&#39;un skill dédié — voir `beta-deployment-availability` dans `coreweave/docs-skills`. Ne la redérivez pas ici.

<div id="8-the-docs-oracle-runs-in-one-direction-only">
  ### 8. L&#39;oracle de la documentation ne fonctionne que dans un seul sens
</div>

La présence de documentation **renforce** la confiance dans le fait qu&#39;une surface est bien active ; une dérive la concernant est donc réelle. Son absence, en revanche, ne doit **jamais** la diminuer : « disponible mais non documenté » est précisément la lacune que l&#39;on cherche à débusquer, et se servir de cette absence pour étouffer un signal referme une boucle dont le détecteur ne sort plus jamais : semble non publié → signal étouffé → personne ne rédige de documentation → toujours pas de documentation → signal toujours étouffé.

Cette règle est appliquée de façon structurelle plutôt que par convention : `docsindex` n&#39;expose aucune fonction renvoyant un score négatif, la boucle est donc impossible à représenter.

À noter également : la correspondance naïve de sous-chaînes ne sert à rien. `search` apparaît sur 215 pages de documentation. Exigez un contexte de mise en évidence dans l&#39;interface utilisateur (`**gras**`, accents graves, guillemets ou « le bouton X »), un filtre de spécificité de ≥ 2 jetons ou en MAJUSCULES, ainsi qu&#39;un plafond sur le nombre de pages.

<div id="9-match-the-literal-case-sensitively-or-you-report-already-fixed-drift">
  ### 9. Faites correspondre le littéral en respectant la casse, sinon vous signalerez une dérive déjà corrigée
</div>

Peu intuitif et facile à prendre à l&#39;envers. Le lookup pose la question : « l&#39;ANCIENNE chaîne
apparaît-elle encore dans la documentation ? » Si la documentation indique `MODELS SEAT` alors que le code indique désormais
`Models Seat`, il y a dérive. Si la documentation indique déjà `Models Seat`, il n&#39;y a
rien à faire. Une correspondance insensible à la casse ne permet pas de distinguer ces deux cas : elle signale donc
la page corrigée comme défectueuse — et le renommage portant uniquement sur la casse est justement le cas
où cela importe le plus.

Les mots environnants (`the`, le nom) peuvent être insensibles à la casse grâce à un
`(?i:...)` circonscrit. Le littéral lui-même ne doit pas l&#39;être.

<div id="10-blank-frontmatter-do-not-delete-it">
  ### 10. Videz le frontmatter, ne le supprimez pas
</div>

Supprimer le frontmatter YAML décale tous les numéros de ligne qui le suivent : une référence
`page:line` ne pointe alors plus vers ce que voit le lecteur — un décalage de cinq lignes, dans notre corpus.
Remplacez-le plutôt par autant de sauts de ligne. C&#39;est peu coûteux, et cela préserve l&#39;exactitude
des citations tout en évitant que les clés du frontmatter soient interprétées comme de la prose.

<div id="11-published-release-notes-are-immutable-and-they-are-a-big-share-of-hits">
  ### 11. Les release notes publiées sont immuables et représentent une grande part des occurrences
</div>

Sur une fenêtre de 60 jours, près de la moitié des occurrences relevées dans la documentation se situent dans `release-notes/**`. Elles constituent une trace historique de ce qui a été livré, sous le nom sous lequel cela a été livré. Les réécrire reviendrait à falsifier un journal des modifications. Signalez-les à titre informatif, ne proposez jamais de modification et ne les comptez jamais dans l&#39;éligibilité des agents.

<div id="12-include-reusable-fragments-exclude-worktrees">
  ### 12. Inclure les fragments réutilisables ; exclure les worktrees
</div>

Deux erreurs de sélection du corpus, de signes opposés :

* **`snippets/`** contient du texte d&#39;interface utilisateur réel (`accédez à l'onglet **Service Accounts**`)
  et alimente le rendu de nombreuses pages : un libellé qui s&#39;y trouve a donc une portée d&#39;impact *bien plus large*
  qu&#39;un libellé présent dans une seule page. L&#39;exclure crée des angles morts.
* **`.claude/`** contient des worktrees git — des copies complètes de l&#39;arborescence. Son indexation
  compte deux fois chaque occurrence et gonfle silencieusement le nombre de pages, ce qui
  déclenche le plafond « trop générique » et masque de véritables résultats.

<div id="13-pair-renames-by-position-before-you-consider-similarity">
  ### 13. Appariez les renommages par position avant d&#39;envisager la similarité
</div>

L&#39;approche évidente — associer une chaîne supprimée à la chaîne ajoutée qui lui
ressemble le plus — échoue précisément dans le cas le plus important. Un libellé
réellement reformulé ne partage presque aucun caractère avec son remplaçant :

| Ancien                      | Nouveau                     | Similarité |
| --------------------------- | --------------------------- | ---------- |
| `Hide manually hidden runs` | `List only visible runs`    | 0.55       |
| `Only show visualized`      | `Hide manually hidden runs` | 0.22       |

Aucun seuil ne capte ces cas sans se mettre à apparier deux en-têtes de colonne
sans rapport. Mais git répond déjà à la question : une modification sur place
apparaît sous la forme d&#39;une ligne `-` et de la ligne `+` qui l&#39;a remplacée, au même
décalage, dans un même bloc de changement. La position est une preuve plus solide que
la similarité de chaînes, et elle n&#39;expose à aucun risque de faux appariement.

La similarité mérite quand même une seconde passe, pour les renommages qui *ne sont pas*
effectués sur place : `header: 'WEAVE ACCESS'` est devenu `name: 'Weave Access'` sur une autre ligne
et dans un autre champ. Regroupez par (path, kind), et non par (path, kind, key), sinon
ce cas-là passera inaperçu.

<div id="14-not-every-conditional-is-a-feature-gate">
  ### 14. Toutes les conditions ne sont pas des feature gates
</div>

Remonter d&#39;une ligne modifiée jusqu&#39;au `if` englobant fait apparaître quantité de blocs qui
n&#39;ont rien à voir avec la visibilité. `if (hideManuallyHidden)` relève de l&#39;état de l&#39;interface utilisateur.
Le signaler comme une gate reviendrait à marquer la moitié de l&#39;application comme « pas encore visible » et à ruiner
la confiance dans le seul signal censé avoir du sens.

Exigez que la variable de la condition se résolve en un hook de gate —
`const shouldShowX = useStatsigGateX(orgName)` — et ne signalez rien dans le cas
contraire. La chaîne est entièrement lisible au sein d&#39;un même diff. La clé Statsig, en revanche,
ne l&#39;est généralement pas : elle réside dans le registre de ramp, traitez-la donc comme un enrichissement
facultatif plutôt que comme une condition préalable.

<div id="15-a-change-to-an-undocumented-label-is-not-drift">
  ### 15. La modification d&#39;un libellé non documenté n&#39;est pas une dérive
</div>

Le premier rapport affichait 22 lignes pour trois commits, dont 2 seulement étaient réelles. Les
autres étaient `new **Loading members**`, `new **Invited**`, `PROFILE removed` — soit toute
chaîne modifiée ne correspondant à aucune page de documentation, chacune consignée comme une « lacune de couverture ».

Une dérive suppose une documentation *par rapport à laquelle* dériver. Renommer un libellé qu&#39;aucune page ne mentionne ne rend
rien incorrect : ce n&#39;est donc pas un constat, tout au plus une statistique. Comptez-les
et affichez le total. Les énumérer noie les lignes qui comptent, or c&#39;est le
seul échec auquel ce rapport ne peut pas survivre : un relecteur qui passe à côté du vrai
constat ne reviendra pas.

Il ne s&#39;agit pas de la suppression que la règle unidirectionnelle interdit. L&#39;absence de documentation
ne doit jamais *masquer un constat existant* ; elle ne doit simplement pas *fabriquer des constats
qui n&#39;existent pas*.

**L&#39;absence de documentation et l&#39;absence de recherche sont deux choses différentes, et le rapport
ne doit pas les confondre.** `docsindex` refuse de rechercher un littéral trop
générique pour être attribué — un mot isolé qui n&#39;est pas en majuscules, donc `Runs`, `Inference`,
`Threshold`. Ceux-là reviennent eux aussi avec zéro occurrence, mais ce zéro signifie « nous
n&#39;avons pas cherché », et non « aucune page ne le mentionne ». Les comptabiliser avec les véritables lacunes de
couverture permettait au rapport d&#39;affirmer que rien dans la documentation n&#39;était devenu faux à propos d&#39;un libellé qu&#39;il
n&#39;avait jamais recherché, ce qui est précisément la violation que le paragraphe ci-dessus
récuse. Ainsi, `build_findings` renvoie deux listes et le rapport affiche deux
titres : **Undocumented surfaces** pour les éléments recherchés et introuvables, **Not attributable**
pour ceux jamais recherchés. C&#39;est le second nombre qu&#39;il faut surveiller : une hausse durable
signifie que `is_specific_enough` absorbe de véritables dérives et demande un réajustement.

Deux cas de figure connexes sont ressortis de la même passe :

* **Agrégez le nouveau texte par surface.** Un nouveau panel de paramètres ajoute un titre, une
  description, deux libellés de champ et un bouton. Cela représente une seule tâche de documentation, pas cinq.
* **Indexez les constats sur la tâche de documentation, pas sur la surface de code.** Trois tableaux de membres
  affichent la même colonne, et la page de documentation la nomme une seule fois. Inclure la surface
  dans l&#39;ID du constat faisait apparaître une seule modification sur trois lignes.

<div id="16-freeze-real-diffs-as-fixtures-immediately">
  ### 16. Figez immédiatement de vrais diffs sous forme de fixtures
</div>

Six sorties `git show` figées dans `tests/fixtures/` constituent à elles seules toute
la surface de régression, et elles ont révélé trois bugs qui avaient survécu à la revue de conception : le JSX inline non détecté,
l&#39;attribut énuméré non détecté et le bug d&#39;identité de déplacement. Aucun
n&#39;apparaissait dans le plan. Les trois ont sauté aux yeux en moins d&#39;une minute d&#39;exécution
sur de vrais diffs.

Lorsque quelqu&#39;un signale un cas non détecté, ajoutez-le comme fixture avant de le corriger.

<div id="17-ownership-is-per-run-data-so-pay-for-it-once">
  ### 17. La propriété est une donnée propre à l&#39;exécution : ne la payez qu&#39;une fois
</div>

Les relecteurs et l&#39;équipe propriétaire ont l&#39;air de recherches effectuées constat par constat, mais n&#39;en sont pas. CODEOWNERS
est un fichier unique qui ne change pas en cours d&#39;exécution, et la paternité du code provient d&#39;un unique
`git log --name-only` sur les racines de l&#39;interface utilisateur, analysé en un index chemin → auteur en
mémoire. La version naïve — un `git show` pour CODEOWNERS plus un ou deux appels `git log`
par constat — revient à environ trois sous-processus par ligne pour des données
identiques sur l&#39;ensemble de l&#39;analyse.

Mettez cela en cache au sein d&#39;une exécution et *non* d&#39;une exécution à l&#39;autre. La composition des équipes évolue, et un cache
de propriétaires obsolète se traduit par une mention @ erronée dans une PR que personne ne saura expliquer.

Les deux réponses sont limitées à une réf., si bien que la clé de cache porte la réf. et que `scan` transmet
son `--head` résolu à `ownership.reset_caches(head=...)`. Lire l&#39;historique
et CODEOWNERS depuis la valeur par défaut configurée alors que la plage de commits provient d&#39;une
autre réf. revient à désigner des personnes qui n&#39;ont jamais touché aux commits de la plage — l&#39;échec
est silencieux, et une mention @ erronée est précisément la sortie qu&#39;un relecteur ne peut pas vérifier
à partir du seul rapport.

L&#39;oracle de documentation mérite le même traitement, pour la même raison. `docsindex.find`
est mémoïsé sur l&#39;index, car le corpus ne change pas en cours d&#39;exécution et les
répétitions sont structurelles : `build_findings` teste un littéral une première fois pour déterminer s&#39;il
est documenté, puis une seconde fois pour y joindre les preuves, et un même libellé change souvent
dans plusieurs commits au sein d&#39;une même fenêtre. Sur 60 jours de `wandb/core`, cela
représente 1180 chaînes modifiées qui se ramènent à un nombre bien plus faible de recherches distinctes.

<div id="18-almost-nothing-needs-to-be-stored-between-runs">
  ### 18. Presque rien n&#39;a besoin d&#39;être conservé entre deux exécutions
</div>

La conception qui vient naturellement à l&#39;esprit pour un détecteur qui ré-analyse tout, c&#39;est un registre qui mémorise chaque
constat déjà émis. Résistez-y. Posez-vous la question pour chaque champ : *une nouvelle analyse
peut-elle recalculer cette valeur ?* Pour l&#39;identité d&#39;un constat, la déduplication, la stabilité, le triage, la responsabilité et
la couverture documentaire, la réponse est oui — toutes les entrées se trouvent dans l&#39;historique des commits ou dans
l&#39;arborescence de la documentation. Les stocker ne fait que créer une seconde copie susceptible de contredire la
première, et c&#39;est justement celle dont personne ne remarque qu&#39;elle est erronée.

Ce qui, en revanche, ne peut réellement pas être recalculé, c&#39;est le fait qu&#39;un humain ait dit « j&#39;ai regardé, tout va bien ». C&#39;est là tout le contenu de `ledger.json`.

La meilleure version de cette astuce ne nécessite aucun fichier. Un projet au même niveau recopie les
release notes du SDK upstream dans une page de documentation et détermine son point de repère en lisant
le `<Update label="...">` le plus récent de la page qu&#39;il maintient — son état *est* l&#39;artifact
publié, si bien que les deux ne peuvent pas diverger. Cherchez d&#39;abord ce schéma-là.

Un corollaire mérite d&#39;être énoncé : dériver plutôt que stocker signifie qu&#39;une nouvelle analyse suffit à corriger une
exécution défaillante. Aucun cache à invalider, aucune migration à écrire lorsqu&#39;un signal
change, et c&#39;est précisément ce qui permet de continuer à faire évoluer les signaux en toute sécurité.

<div id="19-suppression-is-one-directional-too">
  ### 19. La suppression est également unidirectionnelle
</div>

La leçon 8 régit l&#39;oracle de la documentation. La même règle doit régir les décisions stockées, et le mode de défaillance y est plus subtil : un rédacteur écarte un constat en le qualifiant de faux positif et, six semaines plus tard, une page se met à documenter précisément cette surface. Le constat est désormais réel, et le rejet stocké le masquerait silencieusement — une suppression qui devient *de plus en plus* fausse au fil du temps, et que personne ne peut détecter à la lecture du rapport.

Une décision consigne donc les preuves documentaires sur lesquelles elle s&#39;appuie, et tout élargissement de ces preuves la rouvre. Seul l&#39;élargissement compte : si une page cesse de mentionner la chaîne, c&#39;est que quelqu&#39;un a fait le travail, ce qui n&#39;est pas une raison de rouvrir quoi que ce soit.

Les preuves sont des pages et des *nombres d&#39;occurrences par page*, jamais des numéros de ligne. Les numéros de ligne bougent à chaque modification de la documentation, même sans rapport : rouvrir sur cette base ne produirait que du bruit — mais un simple ensemble de noms de pages est trop grossier dans l&#39;autre sens. Il ne permet pas de distinguer une occurrence modifiable sur une page de trois occurrences ; un constat écarté qui gagnait une deuxième occurrence sur une page déjà présente dans l&#39;ensemble restait donc supprimé pendant que la documentation se périmait davantage. Les décomptes se situent entre les deux : insensibles au bruit, sensibles à la croissance. Si vous modifiez à nouveau cette structure, veillez à ce que les anciennes preuves restent équivalentes à un corpus inchangé, sans quoi toutes les décisions stockées se rouvriront d&#39;un coup — le moyen le plus rapide d&#39;apprendre à un rédacteur que le registre crie au loup.

Deux conséquences :

* **Ne supprimez jamais automatiquement une décision.** Une décision orpheline est ambiguë : la dérive a peut-être été résolue, ou la fenêtre d&#39;analyse n&#39;atteint tout simplement pas son commit. Listez-les et laissez un humain choisir.
* **Tenez compte des suppressions dans le rapport.** Un décompte de ce qui a été écarté est le seul moyen pour un lecteur de distinguer « aucune dérive » de « toutes les dérives déjà écartées ». Un détecteur qui laisse tomber des lignes en silence ne peut pas être audité.

<div id="running-it">
  ## Exécution
</div>

```bash
PYTHONPATH=scripts python3 -m uidrift.scan scan --since "60 days ago"
PYTHONPATH=scripts python3 -m uidrift.scan scan --incremental
PYTHONPATH=scripts python3 -m uidrift.scan decide <id> --status dismissed \
    --by matt --agreement false_positive --note "why"
```

`--incremental` prend sa base dans le SHA de tête figurant dans le nom du rapport
le plus récent sous `uidrift/reports/`, ce qui applique la leçon 18 au repère :
les rapports font foi, aucun fichier d&#39;état ne peut donc les contredire.
C&#39;est la différence entre 96 secondes et 1,3 seconde, et c&#39;est ce qui rend une
tâche cron fréquente envisageable.

Les rapports sont nommés `YYYY-MM-DDTHHMMSS-<head sha>.md`, en UTC. L&#39;heure n&#39;est
pas décorative : « le plus récent » a longtemps été déterminé par la seule date, ce
qui conduisait à ordonner deux rapports fusionnés le même jour selon leur SHA de
tête — adressé par contenu, donc de fait aléatoire. Une fois sur deux, c&#39;est le plus
ancien qui est retenu, et un repère qui recule signale à nouveau une dérive
qu&#39;un rédacteur avait déjà écartée. Triez les rapports par nom, jamais par SHA. Les
noms sans horodatage restent analysables et se trient avant ceux du même jour qui
en comportent un, ce qui est le sens sûr : une plage est réanalysée, jamais omise.

L&#39;analyse n&#39;écrit jamais dans le registre ; seul `decide` le fait. `decide` redérive
le constat par une analyse plutôt qu&#39;en lisant un rapport, car l&#39;empreinte des
preuves doit refléter le corpus dans son état actuel — une décision estampillée
de preuves obsolètes ne serait jamais réouverte.

Codes de sortie : `0` tout est propre, `1` erreur de l&#39;opérateur (plage invalide,
checkout manquant), `2` aucune sous-commande, `3` au moins une décision réouverte.
`3` est distinct parce qu&#39;une décision réouverte est le seul résultat qui doive pouvoir
faire échouer une étape de CI.

`--summary-json PATH` écrit les décomptes de l&#39;exécution à l&#39;intention d&#39;un appelant
qui doit prendre une décision. Une étape de CI qui choisit d&#39;ouvrir ou non une PR
devrait lire ce fichier plutôt que d&#39;appliquer un grep au rapport rendu : la prose existe pour être lue, et
coupler un flux de travail à une phrase comme « Aucune dérive à traiter dans cette
fenêtre » rend la formulation porteuse.

<div id="running-it-in-ci">
  ## Exécution dans la CI
</div>

`.github/workflows/uidrift-scan.yml` est le puits. Trois éléments qu&#39;il contient
relèvent de la mesure plutôt que de la préférence : ce sont ceux à conserver lors de l&#39;adaptation.

**Clonez le dépôt surveillé en entier, en branche unique, avec `--no-checkout`.** Chacun
de ces choix a été testé sur `wandb/core` (2,4 Go, 50 000 commits) :

| Clone                       | Coût             | Verdict                     |
| --------------------------- | ---------------- | --------------------------- |
| Complet, `--no-checkout`    | 1,3 Go, ~3,5 min | **Ce que nous utilisons**   |
| `--shallow-since=7 months`  | 136 Mo, ~10 s    | Mauvais relecteurs          |
| `--shallow-since=18 months` | 348 Mo, ~15 s    | Relecteurs toujours erronés |
| `--filter=blob:none`        | 96 Mo, ~17 s     | Inutilisable                |

Le clone superficiel est le choix tentant, et il est erroné pour une raison précise : la propriété
se rabat sur l&#39;ensemble des contributions historiques lorsqu&#39;un fichier compte moins de
`MIN_RECENT_AUTHORS` auteurs récents — soit exactement l&#39;historique dont un clone
tronqué ne dispose pas. Les deux fenêtres ci-dessus ont désigné des relecteurs différents de ceux
issus de l&#39;historique complet, et aucune fenêtre n&#39;est sûre : le mécanisme de repli existe *justement* pour les fichiers anciens et peu modifiés.

Un clone partiel sans blobs paraît idéal (le plus léger, historique de commits complet,
et la propriété n&#39;a besoin que des arborescences), mais `iter_commits` lit `--numstat`, qui nécessite
le contenu des blobs. Une fenêtre d&#39;une journée a passé 82 secondes en récupération paresseuse avant d&#39;échouer sur
le dépôt distant promisor. Si vous parvenez à faire fonctionner ce clone, sachez que `iter_commits` n&#39;utilise jamais les
compteurs d&#39;ajouts/suppressions qu&#39;il analyse — uniquement `cols[2]`, le chemin — et que `--name-only` suffirait donc.
Il s&#39;agit d&#39;une modification de code vendorisé : faites-la en toute connaissance de cause.

**Pas de `[skip ci]` sur le commit.** Le tableau d&#39;anatomie indiquait auparavant le contraire. Un
workflow ignoré ne remonte aucun statut, si bien que les vérifications `pull_request` requises restent
en attente indéfiniment et que la PR ne peut jamais être fusionnée. Faites plutôt en sorte que le rapport coûte peu à la CI :
`.mintignore` exclut `uidrift/`, et Validate MDX ne voit aucun `.mdx` ni `.json` dans
le diff et s&#39;arrête aussitôt.

**Une seule branche glissante, et un commit uniquement lorsqu&#39;il y a quelque chose à dire.** Puisque
le repère est le rapport le plus récent *sur la branche par défaut*, une exécution régénère le
delta depuis le dernier rapport **fusionné** — un rapport supplanté n&#39;a donc plus rien
à dire, et une nouvelle branche par jour ouvré enfouirait la branche courante. Une exécution sans
constat ne commite rien et laisse le repère là où il était ; l&#39;exécution suivante
réanalyse la même plage sur une fenêtre légèrement plus large, ce qui ne coûte que quelques secondes. L&#39;autre
solution — commiter chaque jour un rapport « aucun constat » pour faire avancer le
repère — offre une analyse moins coûteuse au prix d&#39;une PR que personne n&#39;a envie de lire.

<div id="volume-expectations">
  ## Volumes attendus
</div>

Calibrez avant de développer. Pour `wandb/core` sur 60 jours :

| Étape                                              | Nombre                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| Commits sur `origin/master`                        | 2 990                                                               |
| Touchant `frontends/app/src/**/*.tsx`              | 592                                                                 |
| **Candidats de l&#39;étape 1**                     | **170** (~20/semaine)                                               |
| …avec une occurrence dans la documentation publiée | **12** (~1,5/semaine)                                               |
| Réduction                                          | 71 % à l&#39;étape 1 ; 93 % après la jointure avec la documentation |

La jointure avec la documentation est le véritable filtre, et il est déterministe. N&#39;ayez pas recours à un
modèle avant cette étape : évaluer 170 commits coûte un ordre de grandeur de plus que
d&#39;évaluer les 12 qui touchent réellement du contenu publié.

L&#39;analyse complète s&#39;exécute en ~27 secondes, sans réseau ni jeton, car
`gitsource.commit_diff` accepte un pathspec et ne récupère jamais de diffs en dehors des
racines de l&#39;interface utilisateur. Conservez cette propriété : sans elle, un commit de 2 600 lignes devient hors de prix.

Si le nombre de candidats de l&#39;étape 1 dépasse ~250/60 j, resserrez les critères avant d&#39;ajouter l&#39;étape 2 : une passe de
modèle sur l&#39;ensemble du stream `.tsx` relève surtout du gaspillage.

<div id="the-vendored-modules">
  ## Les modules vendorisés
</div>

`_vendor/` contient des copies de `gitsource.py`, `diff_signals.py` et
`commit_text.py` issues de `wandb/release-note-genie`, chacune accompagnée d&#39;un en-tête de provenance
indiquant le commit source. Ce sont volontairement des copies et non des imports : ce
détecteur doit pouvoir s&#39;exécuter dans `wandb/docs` sans dépendre d&#39;un autre dépôt
extrait localement, et ces trois fichiers, stables, n&#39;utilisent que la bibliothèque standard.

Ne re-vendorisez que de manière délibérée, jamais automatiquement.