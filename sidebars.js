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
          'reference-guide/artifacts/intro',
          'reference-guide/artifacts/quickstart',
          'reference-guide/artifacts/construct-an-artifact',
          'reference-guide/artifacts/download-and-use-an-artifact',
          'reference-guide/artifacts/update-an-artifact',
          'reference-guide/artifacts/create-a-custom-alias',
          'reference-guide/artifacts/create-a-new-artifact-version',
          'reference-guide/artifacts/track-external-files',
          'reference-guide/artifacts/delete-artifacts',
          'reference-guide/artifacts/explore-and-traverse-an-artifact-graph',
          'reference-guide/artifacts/storage',
          'reference-guide/artifacts/data-privacy-and-compliance',
          // 'reference-guide/artifacts/artifacts-faqs',
          'reference-guide/artifacts/examples',
        ]
      },
      {
        type: 'category',
        label: 'Tune Hyperparameters',
        items: [
          'reference-guide/tune-hyperparameters/intro',
          'reference-guide/tune-hyperparameters/quickstart',
          'reference-guide/tune-hyperparameters/add-w-and-b-to-your-code',
          'reference-guide/tune-hyperparameters/define-sweep-configuration',
          'reference-guide/tune-hyperparameters/initialize-sweeps',
          'reference-guide/tune-hyperparameters/start-sweep-agents',
          'reference-guide/tune-hyperparameters/parallelize-agents',
          'reference-guide/tune-hyperparameters/visualize-sweep-results',
          'reference-guide/tune-hyperparameters/pause-resume-and-cancel-sweeps',
          'reference-guide/tune-hyperparameters/local-controller',
          'reference-guide/tune-hyperparameters/troubleshoot-sweeps',
          // 'reference-guide/tune-hyperparameters/faq',
          'reference-guide/tune-hyperparameters/useful-resources',
          'reference-guide/tune-hyperparameters/existing-project',
        ]
      },
      {
        type: 'category',
        label: 'Collaborative Reports',
        items: [
          'reference-guide/reports/intro',
          'reference-guide/reports/create-a-report',
          'reference-guide/reports/edit-a-report',
          'reference-guide/reports/collaborate-on-reports',
          'reference-guide/reports/clone-and-export-reports',
          'reference-guide/reports/embed-reports',
          'reference-guide/reports/cross-project-reports',
          'reference-guide/reports/reports-faq',
        ]
      },
      {
        type: 'category',
        label: 'Model Management',
        items: [
          'reference-guide/models/intro',
          'reference-guide/models/model-management-concepts',
          'reference-guide/models/walkthrough',
        ]
      },
      {
        type: 'category',
        label: 'Data Visualization',
        items: [
          'reference-guide/data-vis/intro',
          'reference-guide/data-vis/tables-quickstart',
          'reference-guide/data-vis/log-tables',
          'reference-guide/data-vis/tables',
        ]
      }
  ],
  integrations: [
    'integrations/intro',
  ],
  howtoguides: [
    'howtoguides/intro',
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
