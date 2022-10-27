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
          'reference-guide/artifacts/intro'
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
