---
title: NOTICE DE LICENSE
---

<div id="license-notice-for-reference-documentation-generation">
  # Notice de License pour la génération de la documentation de référence
</div>

<div id="overview">
  ## Aperçu
</div>

Les scripts de ce répertoire servent uniquement à générer la documentation de référence pendant le processus de développement/CI. Ils ne sont PAS distribués avec la bibliothèque Weave et ne font partie d&#39;aucun code de production.

<div id="dependencies-and-their-licenses">
  ## Dépendances et licences associées
</div>

<div id="direct-dependencies">
  ### Dépendances directes
</div>

* **requests** (Apache-2.0) : utilisé pour effectuer des requêtes HTTP
* **lazydocs** (MIT) : générateur de documentation maintenu par W&amp;B

<div id="transitive-dependencies-via-lazydocs">
  ### Dépendances transitives (via lazydocs)
</div>

* **setuptools** (MIT avec des composants LGPL-3.0 intégrés) : Système de build
* Diverses autres dépendances avec des licenses mixtes

<div id="important-notes">
  ## Notes importantes
</div>

1. **Uniquement pour le développement** : ces dépendances ne sont installées que temporairement lors de la génération de la documentation en CI/GitHub Actions. Elles ne sont jamais incluses dans le package Weave distribué.

2. **Aucune distribution** : la documentation générée consiste uniquement en fichiers MDX/Markdown, sans code exécutable ni dépendances.

3. **Exécution isolée** : la GitHub Action exécute ces scripts dans des environnements virtuels isolés qui sont détruits après utilisation.

4. **Conformité aux Licenses** : puisque ces outils ne sont pas distribués avec Weave, les composants LGPL-3.0 présents dans les dépendances embarquées de setuptools ne créent aucune obligation de License pour les utilisateurs de Weave.

<div id="for-organizations-with-strict-license-policies">
  ## Pour les organisations ayant des politiques de License strictes
</div>

Si votre organisation applique des politiques interdisant tout code LGPL dans les outils de développement :

1. Utilisez la GitHub Action pour générer la documentation dans le cloud (recommandé)
2. Utilisez le générateur Python minimal qui évite lazydocs
3. Générez la documentation dans un conteneur Docker
4. Demandez une exception pour les outils utilisés uniquement en développement

<div id="socket-security">
  ## Socket Security
</div>

Le fichier `.socketignore`, à la racine du dépôt, exclut ces scripts de l’analyse de sécurité, car il s’agit d’outils de développement et non de code de production.

<div id="known-socket-security-warnings">
  ### Avertissements connus de Socket Security
</div>

* **Code natif dans wheel** : le package `wheel` contient du code natif, ce qui est normal pour les outils de packaging Python
* **Violations de licence** : certaines dépendances transitives peuvent être soumises à des licences LGPL ou à d&#39;autres licences qui déclenchent des avertissements de conformité aux politiques

Ces avertissements sont acceptables parce que :

1. Les outils sont utilisés uniquement pendant la génération de la documentation
2. Ils s&#39;exécutent dans des environnements CI isolés
3. Ils ne sont jamais distribués avec Weave
4. La documentation générée ne contient aucun code exécutable