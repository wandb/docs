/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  referenceSidebar: [
    'reference-guide/intro',
      {
        type: 'category',
        label: 'Artifacts',
        items: [
          // 'reference-guide/artifacts/artifacts-faqs',
          'reference-guide/artifacts/construct-an-artifact',
          'reference-guide/artifacts/create-a-custom-alias',
          'reference-guide/artifacts/create-a-new-artifact-version',
          'reference-guide/artifacts/data-privacy-and-compliance',
          'reference-guide/artifacts/delete-artifacts',
          'reference-guide/artifacts/download-and-use-an-artifact',
          'reference-guide/artifacts/examples',
          'reference-guide/artifacts/explore-and-traverse-an-artifact-graph',
          'reference-guide/artifacts/intro',
          'reference-guide/artifacts/quickstart',
          'reference-guide/artifacts/storage',
          'reference-guide/artifacts/track-external-files',
          'reference-guide/artifacts/update-an-artifact'
        ]
      },
      {
        type: 'category',
        label: 'Tune Hyperparameters',
        items: [
          'reference-guide/tune-hyperparameters/intro'
        ]
      }
  ],

  api: [
    'api/intro',
    {
      type: 'category',
      label: 'Python Library',
      items: [
        'api/sdk/data-types'
      ]
    }
  ]
};

module.exports = sidebars;
