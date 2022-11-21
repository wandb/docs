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
    'guides/intro',
    'quickstart',
    {
      type: 'category',
      label: 'Experiment Tracking',
      items: [
        'guides/track/intro',
        'guides/track/launch',
        'guides/track/config',
        {
          type: 'category',
          label: 'Log Data with wandb.log',
          items: [
            'guides/track/log/intro',
            'guides/track/log/plots',
            'guides/track/log/log-tables',
            'guides/track/log/working-with-csv',
            'guides/track/log/logging-faqs',
          ]
        },
        'guides/track/advanced/alert',
        'guides/track/app',
        'guides/track/limits',
        'guides/track/public-api-guide',
        'guides/track/jupyter',
        {
          type: 'category',
          label: 'Advanced Features',
          items: [
            'guides/track/advanced/intro',
            'guides/track/advanced/distributed-training',
            'guides/track/advanced/grouping',
            'guides/track/advanced/resuming',
            'guides/track/advanced/save-restore',
            'guides/track/advanced/environment-variables',
          ]
        }
      ]
    },
    {
      type: 'category',
      label: 'Artifacts',
      items: [
        'guides/artifacts/intro',
        'guides/artifacts/quickstart',
        'guides/artifacts/construct-an-artifact',
        'guides/artifacts/download-and-use-an-artifact',
        'guides/artifacts/update-an-artifact',
        'guides/artifacts/create-a-custom-alias',
        'guides/artifacts/create-a-new-artifact-version',
        'guides/artifacts/track-external-files',
        'guides/artifacts/delete-artifacts',
        'guides/artifacts/explore-and-traverse-an-artifact-graph',
        'guides/artifacts/storage',
        'guides/artifacts/data-privacy-and-compliance',
        // 'guides/artifacts/artifacts-faqs',
        'guides/artifacts/examples',
      ]
    },
    {
      type: 'category',
      label: 'Tune Hyperparameters',
      items: [
        'guides/tune-hyperparameters/intro',
        'guides/tune-hyperparameters/quickstart',
        'guides/tune-hyperparameters/add-w-and-b-to-your-code',
        'guides/tune-hyperparameters/define-sweep-configuration',
        'guides/tune-hyperparameters/initialize-sweeps',
        'guides/tune-hyperparameters/start-sweep-agents',
        'guides/tune-hyperparameters/parallelize-agents',
        'guides/tune-hyperparameters/visualize-sweep-results',
        'guides/tune-hyperparameters/pause-resume-and-cancel-sweeps',
        'guides/tune-hyperparameters/sweeps-ui',
        'guides/tune-hyperparameters/local-controller',
        'guides/tune-hyperparameters/troubleshoot-sweeps',
        'guides/tune-hyperparameters/faq',
        'guides/tune-hyperparameters/useful-resources',
        'guides/tune-hyperparameters/existing-project',
      ]
    },
    {
      type: 'category',
      label: 'Collaborative Reports',
      items: [
        'guides/reports/intro',
        'guides/reports/create-a-report',
        'guides/reports/edit-a-report',
        'guides/reports/collaborate-on-reports',
        'guides/reports/clone-and-export-reports',
        'guides/reports/embed-reports',
        'guides/reports/cross-project-reports',
        'guides/reports/reports-faq',
      ]
    },
    {
      type: 'category',
      label: 'Data and model versioning',
      items: [
        'guides/data-and-model-versioning/intro',
        'guides/data-and-model-versioning/dataset-versioning',
        'guides/data-and-model-versioning/model-versioning',
      ]
    },
    {
      type: 'category',
      label: 'Model Management',
      items: [
        'guides/models/intro',
        'guides/models/model-management-concepts',
        'guides/models/walkthrough',
      ]
    },
    {
      type: 'category',
      label: 'Data Visualization',
      items: [
        'guides/data-vis/intro',
        'guides/data-vis/tables-quickstart',
        'guides/data-vis/log-tables',
        'guides/data-vis/tables',
      ]
    },
    {
      type: 'category',
      label: 'W&B App UI',
      items: [
        'guides/app/intro',
        {
          type: 'category',
          label: 'Features',
          items: [
            // 'guides/app/features/intro',
            {
              type: 'category',
              label: 'Panels',
              items: [
                'guides/app/features/panels/intro',
                {
                  type: 'category',
                  label: 'Line Plot',
                  items: [
                    'guides/app/features/panels/line-plot/intro',
                    'guides/app/features/panels/line-plot/reference',
                    'guides/app/features/panels/line-plot/sampling',
                    'guides/app/features/panels/line-plot/smoothing',
                  ]
                },
                'guides/app/features/panels/bar-plot',
                'guides/app/features/panels/run-colors',
                'guides/app/features/panels/parallel-coordinates',
                'guides/app/features/panels/scatter-plot',
                'guides/app/features/panels/code',
                'guides/app/features/panels/parameter-importance',
                'guides/app/features/panels/run-comparer',
                {
                  type: 'category',
                  label: 'Weave',
                  items: [
                    'guides/app/features/panels/weave/intro',
                    'guides/app/features/panels/weave/embedding-projector',
                  ]
                },
              ]
            },
            {
              type: 'category',
              label: 'Custom Charts',
              items: [
                'guides/app/features/custom-charts/intro',
                'guides/app/features/custom-charts/walkthrough',
              ]
            },
            'guides/app/features/runs-table',
            'guides/app/features/tags',
            'guides/app/features/notes',
            // 'guides/app/features/alerts',
            'guides/app/features/teams',
            'guides/app/features/system-metrics',
            'guides/app/features/anon',
          ]
        },
        {
          type: 'category',
          label: 'Pages',
          items: [
            'guides/app/pages/intro',
            'guides/app/pages/gradient-panel',
            'guides/app/pages/project-page',
            'guides/app/pages/run-page',
            'guides/app/pages/workspaces',
          ]
        },
        {
          type: 'category',
          label: 'Settings',
          items: [
            'guides/app/settings-page/intro',
            'guides/app/settings-page/user-settings',
            'guides/app/settings-page/team-settings',
            'guides/app/settings-page/emails',
          ]
        },
      ]
    },
    {
      type: 'category',
      label: 'Private Hosting',
      items: [
        'guides/hosting/intro',
        'guides/hosting/basic-setup',
        {
          type: 'category',
          label: 'Production Setup',
          items: [
            'guides/hosting/setup/intro',
            'guides/hosting/setup/dedicated-cloud',
            'guides/hosting/setup/private-cloud',
            'guides/hosting/setup/on-premise-baremetal',
            'guides/hosting/setup/configuration',
          ]
        },
        'guides/hosting/faq',
      ]
    },
    {
      type: 'category',
      label: 'Technical FAQ',
      items: [
        // 'guides/hosting/technical-faq/intro',
        'guides/technical-faq/general',
        'guides/technical-faq/metrics-and-performance',
        'guides/technical-faq/setup',
        'guides/technical-faq/troubleshooting',
      ]
    },
    {
      type: 'category',
      label: 'Integrations',
      items: [
        'guides/integrations/intro',
        'guides/integrations/keras',
        'guides/integrations/pytorch',
        'guides/integrations/lightning',
        'guides/integrations/tensorflow',
        'guides/integrations/tensorboard',
        'guides/integrations/huggingface',
        'guides/integrations/spacy',
        'guides/integrations/yolov5',
        'guides/integrations/mmdetection',
        'guides/integrations/scikit',
        'guides/integrations/xgboost',
        'guides/integrations/lightgbm',
        {
          type: 'category',
          label: 'Fastai',
          items: [
            'guides/integrations/fastai/intro',
            'guides/integrations/fastai/v1',
          ]
        },
        'guides/integrations/other/catalyst',
        'guides/integrations/other/databricks',
        'guides/integrations/other/deepchecks',
        'guides/integrations/other/deepchem',
        'guides/integrations/other/docker',
        'guides/integrations/other/hydra',
        'guides/integrations/other/kubeflow-pipelines-kfp',
        'guides/integrations/other/metaflow',
        'guides/integrations/other/mmf',
        'guides/integrations/other/composer',
        'guides/integrations/other/openai',
        'guides/integrations/other/openai-gym',
        'guides/integrations/other/paddledetection',
        'guides/integrations/other/paddleocr',
        'guides/integrations/other/prodigy',
        'guides/integrations/other/ignite',
        'guides/integrations/other/ray-tune',
        'guides/integrations/other/sagemaker',
        'guides/integrations/other/simpletransformers',
        'guides/integrations/other/skorch',
        'guides/integrations/other/stable-baselines-3',
        'guides/integrations/other/w-and-b-for-julia',
        'guides/integrations/other/yolox',
      ]
    },
  ],
  howtoguides: [
    'howtoguides/intro',
  ],
  ref: [
    'ref/intro',
    {
      type: 'category',
      label: 'Python Library',
      items: [
        'ref/python/README',
        'ref/python/artifact'
      ]
    },
    {
      type: 'category',
      label: 'Command Line Interface',
      items: [
        'ref/cli/README'
      ]
    }
  ]
};

module.exports = sidebars;
