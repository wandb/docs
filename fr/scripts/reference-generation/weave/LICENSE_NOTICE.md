---
title: MENTION DE LICENCE
---

# Notice de licence relative à la génération de la documentation de référence {#license-notice-for-reference-documentation-generation}

## Aperçu {#overview}

Les scripts de ce répertoire servent uniquement à générer la documentation de référence dans le cadre du processus de développement/CI. Ils ne sont PAS distribués avec la bibliothèque Weave ni inclus dans aucun code de production.

## Dépendances et licences associées {#dependencies-and-their-licenses}

### Dépendances directes {#direct-dependencies}

* **requests** (Apache-2.0) : Utilisé pour les requêtes HTTP
* **lazydocs** (MIT) : Génateur de documentation maintenu par W&amp;B

### Dépendances transitives (via lazydocs) {#transitive-dependencies-via-lazydocs}

* **setuptools** (MIT avec des composants LGPL-3.0 intégrés) : système de compilation
* Diverses autres dépendances sous différentes licences

## Notes importantes {#important-notes}

1. **Développement uniquement** : ces dépendances sont installées uniquement de façon temporaire lors de la génération de la documentation dans CI/GitHub Actions. Elles ne sont jamais incluses dans le package Weave distribué.

2. **Aucune distribution** : la documentation générée se compose uniquement de fichiers MDX/Markdown, sans code exécutable ni dépendances.

3. **Exécution isolée** : la GitHub Action exécute ces scripts dans des environnements virtuels isolés, qui sont supprimés après utilisation.

4. **Conformité des licences** : puisque ces outils ne sont pas distribués avec Weave, les composants LGPL-3.0 présents dans les dépendances intégrées de setuptools n&#39;entraînent pas d&#39;obligations de licence pour les utilisateurs de Weave.

## Pour les organisations ayant des politiques de licence strictes {#for-organizations-with-strict-license-policies}

Si votre organisation applique des politiques interdisant tout code LGPL dans les outils de développement :

1. Utilisez la GitHub Action pour générer la documentation dans le cloud (recommandé)
2. Utilisez le générateur Python minimal qui évite lazydocs
3. Générez la documentation dans un conteneur Docker
4. Demandez une exception pour les outils réservés au développement

## Socket Security {#socket-security}

Le fichier `.socketignore`, situé à la racine du dépôt, exclut ces scripts de l’analyse de sécurité, car il s’agit d’outils de développement et non de code de production.

### Avertissements Socket Security connus {#known-socket-security-warnings}

* **Code natif dans `wheel`** : le package `wheel` contient du code natif, ce qui est normal pour les outils de packaging Python.
* **Violations de licence** : certaines dépendances transitives peuvent être soumises à des licences LGPL ou à d&#39;autres licences qui déclenchent des avertissements de politique.

Ces avertissements sont acceptables pour les raisons suivantes :

1. Les outils sont utilisés uniquement lors de la génération de la documentation.
2. Ils s&#39;exécutent dans des environnements de CI isolés.
3. Ils ne sont jamais distribués avec Weave.
4. La documentation générée ne contient aucun code exécutable.