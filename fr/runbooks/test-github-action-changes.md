---
title: Tester les modifications de GitHub Actions
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # Prompt de l’agent : tester les changements apportés à GitHub Actions dans wandb/docs
</div>

<div id="requirements">
  ## Prérequis
</div>

* **Accès employé W&amp;B** : vous devez être employé chez W&amp;B et avoir accès aux systèmes internes de W&amp;B.
* **Fork GitHub** : un fork personnel de wandb/docs pour tester les modifications du flux de travail. Dans ce fork, vous devez avoir l’autorisation de pousser vers la branche par défaut et de contourner les règles de protection des branches.

<div id="agent-prerequisites">
  ## Prérequis de l’agent
</div>

Avant de commencer, rassemblez les informations suivantes :

1. **Nom d’utilisateur GitHub** - Vérifiez d’abord `git remote -v` pour identifier le dépôt distant du fork, puis `git config` pour le nom d’utilisateur. Ne le demandez à l’utilisateur que s’il est introuvable dans les deux emplacements.
2. **Statut du fork** - Vérifiez qu’il dispose d’un fork de wandb/docs avec l’autorisation de pousser vers la branche par défaut et de contourner la protection de branche.
3. **Portée du test** - Demandez quelles modifications précises sont testées (mise à niveau de dépendances, modification de fonctionnalité, etc.).

<div id="task-overview">
  ## Aperçu de la tâche
</div>

Testez les modifications apportées aux flux de travail de GitHub Actions dans le dépôt wandb/docs.

<div id="context-and-constraints">
  ## Contexte et contraintes
</div>

<div id="repository-setup">
  ### Configuration du dépôt
</div>

* **Dépôt principal** : `wandb/docs` (origin)
* **Fork pour les tests** : `<username>/docs` (remote du fork) - Si ce n’est pas clair avec `git remoter -v`, demandez à l’utilisateur l’endpoint de son fork.
* **Important** : les GitHub Actions dans les PR s’exécutent toujours depuis la branche de base (main), et non depuis la branche de la PR.
* **Limitation de déploiement Mintlify** : les déploiements Mintlify et la vérification `link-rot` ne se buildent que pour le dépôt principal wandb/docs, pas pour les forks. Dans un fork, la GitHub Action `validate-mdx` vérifie le statut des commandes `mint dev` et `mint broken-links` dans une PR du fork.

**Note de l’agent** : Vous devez :

1. Vérifier `git remote -v` pour voir s’il existe déjà un remote de fork et extraire le nom d’utilisateur de l’URL s’il est présent.
2. Si le nom d’utilisateur n’est pas trouvé dans les remotes, vérifier `git config` pour le nom d’utilisateur GitHub.
3. Ne demander à l’utilisateur son nom d’utilisateur GitHub que s’il n’est trouvé à aucun de ces emplacements.
4. Vérifier qu’il dispose d’un fork de wandb/docs pouvant être utilisé pour les tests.
5. Si vous ne pouvez pas pousser directement vers le fork, créez une branche temporaire dans wandb/docs pour que l’utilisateur puisse pousser depuis celle-ci.

<div id="testing-requirements">
  ### Exigences de test
</div>

Pour tester les modifications des flux de travail, vous devez :

1. Synchroniser le `main` du fork avec le `main` du dépôt principal, en supprimant tous les commits temporaires.
2. Appliquer les modifications à la branche `main` du fork (pas uniquement à une branche de fonctionnalité)
3. Créer une PR de test sur le `main` du fork, avec des modifications de contenu pour déclencher les flux de travail.

<div id="step-by-step-testing-process">
  ## Processus de test étape par étape
</div>

<div id="1-initial-setup">
  ### 1. Configuration initiale
</div>

```bash
# Vérifier les remotes existants
git remote -v

# Si le remote fork existe, noter le nom d'utilisateur depuis l'URL du fork
# Si le remote fork est manquant, vérifier git config pour le nom d'utilisateur
git config user.name  # ou git config github.user

# Ne demander à l'utilisateur son nom d'utilisateur GitHub ou les détails du fork que s'ils sont introuvables dans les remotes ou la configuration
# Exemple de question : « Quel est votre nom d'utilisateur GitHub pour le fork que nous utiliserons pour les tests ? »

# Si le remote fork est manquant, l'ajouter :
git remote add fork https://github.com/<username>/docs.git  # Remplacer <username> par le nom d'utilisateur réel
```


<div id="2-sync-fork-and-prepare-test-branch">
  ### 2. Synchroniser le fork et préparer la branche de test
</div>

```bash
# Récupérer les dernières modifications depuis origin
git fetch origin

# Basculer sur main et réinitialiser à origin/main pour assurer une synchronisation propre
git checkout main
git reset --hard origin/main

# Forcer le push vers le fork pour le synchroniser (en supprimant les commits temporaires du fork)
git push fork main --force

# Créer une branche de test pour les modifications du flux de travail
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. Appliquez les modifications au flux de travail
</div>

Apportez vos modifications aux fichiers du flux de travail. Pour les mises à niveau des dépendances :

* Mettez à jour les numéros de version dans les instructions `uses:`
* Vérifiez les deux fichiers de flux de travail si la dépendance est utilisée à plusieurs endroits

**Astuce de pro** : Avant de finaliser un runbook, demandez à un agent IA de le passer en revue avec un prompt comme :

> &quot;Veuillez examiner ce runbook et suggérer des améliorations pour le rendre plus utile aux agents IA. Concentrez-vous sur la clarté, l&#39;exhaustivité et l&#39;élimination des ambiguïtés.&quot;

<div id="5-commit-and-push-to-forks-main">
  ### 5. Validez et poussez vers la branche `main` du fork
</div>

```bash
# Valider toutes les modifications
git add -A
git commit -m "test: [Description of what you're testing]"

# Pousser vers la branche `main` du fork
git push fork HEAD:main --force-with-lease
```

**Instructions de l’agent pour l’accès au fork** :
Si vous ne pouvez pas pousser directement sur le fork :

1. Créez une branche temporaire dans wandb/docs avec les modifications
2. Donnez à l’utilisateur cette commande :
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. Indiquez-lui de créer la PR à l’adresse suivante : `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. N’oubliez pas de supprimer la branche temporaire de wandb/docs après les tests


<div id="6-create-test-pr">
  ### 6. Créer une PR de test
</div>

```bash
# Créer une nouvelle branche à partir du fork main mis à jour
git checkout -b test-pr-[description]

# Effectuer une petite modification de contenu pour déclencher les flux de travail
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# Commiter et pousser
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

Créez ensuite une PR depuis l’UI GitHub de `<username>:test-pr-[description]` vers `<username>:main`


<div id="7-monitor-and-verify">
  ### 7. Surveiller et vérifier
</div>

Comportement attendu :

1. Le bot GitHub Actions crée un commentaire initial avec « Generating preview links... »
2. Le flux de travail doit se terminer sans erreur

À vérifier :

* ✅ Le flux de travail se termine correctement
* ✅ Le commentaire de prévisualisation est créé puis mis à jour
* ✅ Les liens utilisent l’URL de remplacement
* ✅ La catégorisation des fichiers fonctionne (Ajouté/Modifié/Supprimé/Renommé)
* ❌ La moindre erreur dans les journaux GitHub Actions
* ❌ Des avertissements de sécurité ou des secrets exposés

<div id="8-cleanup">
  ### 8. Nettoyage
</div>

Après les tests :

```bash
# Réinitialiser la branche main du fork pour correspondre à l'upstream
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# Supprimer les branches de test du fork et de l'origin
git branch -D test-[description]-[date] test-pr-[description]
```


<div id="common-issues-and-solutions">
  ## Problèmes courants et solutions
</div>

<div id="issue-permission-denied-when-pushing-to-fork">
  ### Problème : permission refusée lors d’un push vers un fork
</div>

* Le jeton GitHub est peut-être en lecture seule
* Solution : utilisez SSH ou effectuez le push manuellement depuis votre machine locale

<div id="issue-workflows-not-triggering">
  ### Problème : les flux de travail ne se déclenchent pas
</div>

* N&#39;oubliez pas : les flux de travail s&#39;exécutent à partir de la branche de base (`main`), et non de la branche de la PR
* Assurez-vous que les modifications se trouvent dans la branche `main` du fork

<div id="issue-changed-files-not-detected">
  ### Problème : fichiers modifiés non détectés
</div>

* Assurez-vous que les modifications de contenu se trouvent dans des répertoires suivis par Git (`content/`, `static/`, `assets/`, etc.)
* Vérifiez le filtre `files:` dans la configuration du flux de travail

<div id="testing-checklist">
  ## Liste de vérification des tests
</div>

* [ ] Demandé à l&#39;utilisateur son nom d&#39;utilisateur GitHub et les détails de son fork
* [ ] Les deux remote (origin et fork) sont configurés
* [ ] Les modifications du flux de travail ont été appliquées aux deux fichiers concernés
* [ ] Les modifications ont été poussées vers la branche `main` du fork (directement ou par l&#39;utilisateur)
* [ ] PR de test créée avec des modifications de contenu
* [ ] Commentaire de prévisualisation généré avec succès
* [ ] Aucune erreur dans les journaux de GitHub Actions
* [ ] Branche `main` du fork réinitialisée après les tests
* [ ] Branches temporaires supprimées de wandb/docs (si elles ont été créées)