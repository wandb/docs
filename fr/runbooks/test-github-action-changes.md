---
title: Tester les modifications apportées à GitHub Action
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # Prompt de l’agent : tester les modifications de GitHub Actions dans wandb/docs
</div>

<div id="requirements">
  ## Prérequis
</div>

* **Accès employé W&amp;B** : vous devez être employé(e) chez W&amp;B et avoir accès aux systèmes internes de W&amp;B.
* **Fork GitHub** : un fork personnel de wandb/docs pour tester les modifications du workflow. Dans ce fork, vous devez être autorisé à pousser vers la branche par défaut et à contourner les règles de protection de la branche.

<div id="agent-prerequisites">
  ## Prérequis de l&#39;agent
</div>

Avant de commencer, rassemblez les informations suivantes :

1. **Nom d&#39;utilisateur GitHub** - Vérifiez d&#39;abord `git remote -v` pour le remote du fork, puis `git config` pour le nom d&#39;utilisateur. Ne demandez à l&#39;utilisateur que si ces informations sont introuvables dans l&#39;un ou l&#39;autre emplacement.
2. **Statut du fork** - Confirmez qu&#39;il dispose d&#39;un fork de wandb/docs avec l&#39;autorisation de pousser vers la branche par défaut et de contourner la protection de branche.
3. **Portée du test** - Demandez quelles modifications précises sont en cours de test (mise à niveau de dépendance, changement de fonctionnalité, etc.).

<div id="task-overview">
  ## Aperçu de la tâche
</div>

Testez les modifications apportées aux workflows GitHub Actions dans le dépôt wandb/docs.

<div id="context-and-constraints">
  ## Contexte et contraintes
</div>

<div id="repository-setup">
  ### Configuration du dépôt
</div>

* **Dépôt principal** : `wandb/docs` (origin)
* **Fork de test** : `<username>/docs` (remote du fork) - Si ce n’est pas clair à partir de `git remoter -v`, demandez à l’utilisateur l’URL de son fork.
* **Important** : les GitHub Actions des PR s’exécutent toujours à partir de la branche de base (main), et non de la branche de la PR.
* **Limitation de déploiement Mintlify** : les déploiements Mintlify et la vérification `link-rot` ne sont générés que pour le dépôt principal `wandb/docs`, pas pour les forks. Dans un fork, la GitHub Action `validate-mdx` vérifie l’état des commandes `mint dev` et `mint broken-links` dans une PR du fork.

**Note de l’agent** : vous devez :

1. Vérifier `git remote -v` pour voir si un remote de fork existe déjà et extraire le nom d’utilisateur à partir de l’URL, le cas échéant.
2. Si le nom d’utilisateur n’est pas trouvé dans les remotes, vérifiez `git config` pour trouver le nom d’utilisateur GitHub.
3. Ne demandez le nom d’utilisateur GitHub à l’utilisateur que s’il n’est trouvé à aucun de ces emplacements.
4. Vérifiez qu’il dispose d’un fork de `wandb/docs` utilisable pour les tests.
5. Si vous ne pouvez pas pousser directement vers le fork, créez une branche temporaire dans `wandb/docs` afin que l’utilisateur puisse pousser depuis celle-ci.

<div id="testing-requirements">
  ### Exigences de test
</div>

Pour tester les modifications apportées au workflow, vous devez :

1. Synchroniser le `main` du fork avec le `main` du dépôt principal, en supprimant tous les commits temporaires.
2. Appliquer les modifications à la branche `main` du fork (et pas seulement à une branche de fonctionnalité)
3. Créer une PR de test sur le `main` du fork avec des modifications de contenu afin de déclencher les workflows.

<div id="step-by-step-testing-process">
  ## Procédure de test étape par étape
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

# Ne demander le nom d'utilisateur GitHub ou les détails du fork que s'ils sont introuvables dans les remotes ou la config
# Exemple de question : « Quel est votre nom d'utilisateur GitHub pour le fork que nous utiliserons pour les tests ? »

# Si le remote fork est manquant, l'ajouter :
git remote add fork https://github.com/<username>/docs.git  # Remplacer <username> par le nom d'utilisateur réel
```


<div id="2-sync-fork-and-prepare-test-branch">
  ### 2. Synchronisez le fork et préparez la branche de test
</div>

```bash
# Récupérer les dernières modifications depuis origin
git fetch origin

# Basculer sur main et effectuer un hard reset vers origin/main pour garantir une synchronisation propre
git checkout main
git reset --hard origin/main

# Forcer le push vers le fork pour le synchroniser (en supprimant les commits temporaires du fork)
git push fork main --force

# Créer une branche de test pour les modifications du workflow
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. Appliquer les modifications au workflow
</div>

Apportez vos modifications aux fichiers de workflow. Pour les mises à niveau de dépendances :

* Mettez à jour les numéros de version dans les déclarations `uses:`
* Vérifiez les deux fichiers de workflow si la dépendance est utilisée à plusieurs endroits

**Conseil de pro** : Avant de finaliser un runbook, demandez à un agent IA de le relire avec une invite comme :

> &quot;Veuillez relire ce runbook et suggérer des améliorations pour le rendre plus utile aux agents IA. Concentrez-vous sur la clarté, l&#39;exhaustivité et la suppression des ambiguïtés.&quot;

<div id="5-commit-and-push-to-forks-main">
  ### 5. Validez et poussez les modifications sur la branche `main` du fork
</div>

```bash
# Valider toutes les modifications
git add -A
git commit -m "test: [Description of what you're testing]"

# Pousser vers la branche principale du fork
git push fork HEAD:main --force-with-lease
```

**Instructions de l’agent pour l’accès au fork** :
Si vous ne pouvez pas effectuer directement un push vers le fork :

1. Créez une branche temporaire dans wandb/docs avec les modifications
2. Fournissez à l’utilisateur la commande suivante :
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. Indiquez-lui de créer la PR à l’adresse suivante : `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. N’oubliez pas de supprimer la branche temporaire de wandb/docs après le test


<div id="6-create-test-pr">
  ### 6. Créer une PR de test
</div>

```bash
# Créer une nouvelle branche à partir du fork main mis à jour
git checkout -b test-pr-[description]

# Apporter une petite modification de contenu pour déclencher les workflows
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# Commiter et pousser les modifications
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

Créez ensuite une PR dans l’interface GitHub, de `<username>:test-pr-[description]` vers `<username>:main`


<div id="7-monitor-and-verify">
  ### 7. Surveiller et vérifier
</div>

Comportement attendu :

1. Le bot GitHub Actions crée un commentaire initial avec « Generating preview links... »
2. Le workflow doit se terminer sans erreur

Vérifiez les points suivants :

* ✅ Le workflow se termine correctement
* ✅ Le commentaire de prévisualisation est créé puis mis à jour
* ✅ Les liens utilisent l’URL de remplacement
* ✅ La catégorisation des fichiers fonctionne (Added/Modified/Deleted/Renamed)
* ❌ Toute erreur dans les logs d’Actions
* ❌ Des avertissements de sécurité ou des secrets exposés

<div id="8-cleanup">
  ### 8. Nettoyage
</div>

Une fois les tests terminés :

```bash
# Réinitialiser la branche main du fork pour qu'elle corresponde à l'upstream
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
  ### Problème : permission refusée lors du push vers un fork
</div>

* Le jeton GitHub est peut-être en lecture seule.
* Solution : utilisez SSH ou effectuez le push manuellement depuis votre machine locale.

<div id="issue-workflows-not-triggering">
  ### Problème : les workflows ne se déclenchent pas
</div>

* Rappel : les workflows s’exécutent depuis la branche de base (`main`), et non depuis la branche de PR
* Assurez-vous que les modifications se trouvent dans la branche `main` du fork

<div id="issue-changed-files-not-detected">
  ### Problème : les fichiers modifiés ne sont pas détectés
</div>

* Assurez-vous que les modifications du contenu se trouvent dans des répertoires suivis (content/, static/, assets/, etc.)
* Vérifiez le filtre `files:` dans la configuration du workflow

<div id="testing-checklist">
  ## Checklist de test
</div>

* [ ] Demander à l&#39;utilisateur son nom d&#39;utilisateur GitHub et les détails de son fork
* [ ] Configurer les deux remotes (`origin` et le fork)
* [ ] Appliquer les modifications du workflow aux deux fichiers concernés
* [ ] Pousser les modifications vers la branche `main` du fork (directement ou via l&#39;utilisateur)
* [ ] Créer une PR de test avec des modifications de contenu
* [ ] Générer correctement le commentaire de prévisualisation
* [ ] Vérifier l&#39;absence d&#39;erreurs dans les journaux GitHub Actions
* [ ] Réinitialiser la branche `main` du fork après les tests
* [ ] Nettoyer les branches temporaires dans wandb/docs (si elles ont été créées)