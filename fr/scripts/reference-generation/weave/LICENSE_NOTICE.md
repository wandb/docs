---
title: NOTICE DE LICENCE
---

<div id="license-notice-for-reference-documentation-generation">
  # Notice de licence pour la génération de la documentation de référence
</div>

<div id="overview">
  ## Vue d’ensemble
</div>

Les scripts de ce répertoire servent uniquement à générer la documentation de référence pendant le processus de développement/CI. Ils ne sont PAS distribués avec la bibliothèque Weave et ne font partie d’aucun code de production.

<div id="dependencies-and-their-licenses">
  ## Dépendances et licences associées
</div>

<div id="direct-dependencies">
  ### Dépendances directes
</div>

* **requests** (Apache-2.0) : utilisé pour les requêtes HTTP
* **lazydocs** (MIT) : générateur de documentation maintenu par W&amp;B

<div id="transitive-dependencies-via-lazydocs">
  ### Dépendances transitives (via lazydocs)
</div>

* **setuptools** (MIT avec des composants LGPL-3.0 embarqués) : système de compilation
* Diverses autres dépendances avec des licences variées

<div id="important-notes">
  ## Notes importantes
</div>

1. **Développement uniquement** : ces dépendances ne sont installées que temporairement lors de la génération de la documentation dans CI/GitHub Actions. Elles ne sont jamais incluses dans le package Weave distribué.

2. **Aucune distribution** : la documentation générée se compose uniquement de fichiers MDX/Markdown, sans code exécutable ni dépendance.

3. **Exécution isolée** : la GitHub Action exécute ces scripts dans des environnements virtuels isolés, qui sont détruits après utilisation.

4. **Conformité des licences** : puisque ces outils ne sont pas distribués avec Weave, les composants LGPL-3.0 présents dans les dépendances embarquées de setuptools n’imposent aucune obligation de licence aux utilisateurs de Weave.

<div id="for-organizations-with-strict-license-policies">
  ## Pour les organisations soumises à des politiques de licence strictes
</div>

Si votre organisation applique des politiques interdisant tout code LGPL dans les outils de développement :

1. Utilisez la GitHub Action pour générer la documentation dans le cloud (recommandé)
2. Utilisez le générateur Python minimal qui n’utilise pas lazydocs
3. Générez la documentation dans un conteneur Docker
4. Demandez une exception pour les outils utilisés uniquement pour le développement

<div id="socket-security">
  ## Socket Security
</div>

Le fichier `.socketignore`, à la racine du dépôt, exclut ces scripts de l’analyse de sécurité, car il s’agit d’outils de développement et non de code de production.

<div id="known-socket-security-warnings">
  ### Avertissements Socket Security connus
</div>

* **Code natif dans wheel** : le package `wheel` contient du code natif, ce qui est normal pour les outils de packaging Python
* **Violations de licence** : certaines dépendances transitives peuvent être soumises à la licence LGPL ou à d&#39;autres licences qui déclenchent des avertissements de conformité

Ces avertissements sont acceptables pour les raisons suivantes :

1. Ces outils sont utilisés uniquement lors de la génération de la documentation
2. Ils s&#39;exécutent dans des environnements CI isolés
3. Ils ne sont jamais distribués avec Weave
4. La documentation générée ne contient aucun code exécutable