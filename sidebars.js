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
      label: 'Track Experiments',
      link: {type: 'doc', id: 'guides/track/intro'},
      items: [
        // 'guides/track/intro',
        'guides/track/launch',
        'guides/track/config',
        {
          type: 'category',
          label: 'Log Objects and Media',
          link: {type: 'doc', id: 'guides/track/log/intro'},
          items: [
            // 'guides/track/log/intro',
            'guides/track/log/plots',
            'guides/track/log/log-tables',
            'guides/track/log/log-summary',
            'guides/track/log/media',
            'guides/track/log/working-with-csv',
            'guides/track/log/distributed-training',
            'guides/track/log/customize-logging-axes',
            'guides/track/log/logging-faqs',
          ],
        },
        'guides/track/app',
        // 'guides/track/reproduce-experiments',
        'guides/track/jupyter',
        'guides/track/limits',
        'guides/track/public-api-guide',
        'guides/track/tracking-faq',
        'guides/track/save-restore',
        'guides/track/environment-variables',
      ],
    },
    {
      type: 'category',
      label: 'Runs',
      link: {type: 'doc', id: 'guides/runs/intro'},
      items: [
        // 'guides/runs/intro',
        // 'guides/runs/create-run',
        'guides/runs/grouping',
        'guides/runs/resuming',
        'guides/runs/alert',
        'guides/runs/manage-runs',
      ],
    },
    {
      type: 'category',
      label: 'Artifacts',
      link: {type: 'doc', id: 'guides/artifacts/intro'},
      items: [
        // 'guides/artifacts/intro',
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
        'guides/artifacts/artifacts-faqs',
        // 'guides/artifacts/examples',
      ],
    },
    {
      type: 'category',
      label: 'Tune Hyperparameters',
      link: {type: 'doc', id: 'guides/sweeps/intro'},
      items: [
        // 'guides/sweeps/intro',
        'guides/sweeps/quickstart',
        'guides/sweeps/add-w-and-b-to-your-code',
        'guides/sweeps/define-sweep-configuration',
        'guides/sweeps/initialize-sweeps',
        'guides/sweeps/start-sweep-agents',
        'guides/sweeps/parallelize-agents',
        'guides/sweeps/visualize-sweep-results',
        'guides/sweeps/pause-resume-and-cancel-sweeps',
        'guides/sweeps/sweeps-ui',
        'guides/sweeps/local-controller',
        'guides/sweeps/troubleshoot-sweeps',
        'guides/sweeps/faq',
        'guides/sweeps/useful-resources',
        'guides/sweeps/existing-project',
      ],
    },
    {
      type: 'category',
      label: 'Collaborative Reports',
      link: {type: 'doc', id: 'guides/reports/intro'},
      items: [
        // 'guides/reports/intro',
        'guides/reports/create-a-report',
        'guides/reports/edit-a-report',
        'guides/reports/collaborate-on-reports',
        'guides/reports/clone-and-export-reports',
        'guides/reports/embed-reports',
        'guides/reports/cross-project-reports',
        'guides/reports/reports-faq',
      ],
    },
    {
      type: 'category',
      label: 'Launch jobs',
      link: {
        type: 'doc', 
        id : 'guides/launch/intro'
      },
      items: [
        'guides/launch/getting-started',
        'guides/launch/prerequisites',
        'guides/launch/create-job',
        'guides/launch/add-jobs-to-queue',
        'guides/launch/launch-jobs',
      ],
    },
    {
      type: 'category',
      label: 'Data and model versioning',
      link: {type: 'doc', id: 'guides/data-and-model-versioning/intro'},
      items: [
        // 'guides/data-and-model-versioning/intro',
        'guides/data-and-model-versioning/dataset-versioning',
        'guides/data-and-model-versioning/model-versioning',
      ],
    },
    {
      type: 'category',
      label: 'Model Management',
      link: {type: 'doc', id: 'guides/models/intro'},
      items: [
        // 'guides/models/intro',
        'guides/models/model-management-concepts',
        'guides/models/walkthrough',
      ],
    },
    {
      type: 'category',
      label: 'Data Visualization',
      link: {type: 'doc', id: 'guides/data-vis/intro'},
      items: [
        // 'guides/data-vis/intro',
        'guides/data-vis/tables-quickstart',
        'guides/data-vis/tables',
      ],
    },
    {
      type: 'category',
      label: 'W&B App UI',
      link: {type: 'doc', id: 'guides/app/intro'},
      items: [
        // 'guides/app/intro',
        {
          type: 'category',
          label: 'Features',
          link: {type: 'doc', id: 'guides/app/features/intro'},
          items: [
            // 'guides/app/features/intro',
            {
              type: 'category',
              label: 'Panels',
              link: {type: 'doc', id: 'guides/app/features/panels/intro'},
              items: [
                // 'guides/app/features/panels/intro',
                {
                  type: 'category',
                  label: 'Line Plot',
                  link: {
                    type: 'doc',
                    id: 'guides/app/features/panels/line-plot/intro',
                  },
                  items: [
                    // 'guides/app/features/panels/line-plot/intro',
                    'guides/app/features/panels/line-plot/reference',
                    'guides/app/features/panels/line-plot/sampling',
                    'guides/app/features/panels/line-plot/smoothing',
                  ],
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
                  link: {
                    type: 'doc',
                    id: 'guides/app/features/panels/weave/intro',
                  },
                  items: [
                    // 'guides/app/features/panels/weave/intro',
                    'guides/app/features/panels/weave/embedding-projector',
                  ],
                },
              ],
            },
            {
              type: 'category',
              label: 'Custom Charts',
              link: {
                type: 'doc',
                id: 'guides/app/features/custom-charts/intro',
              },
              items: [
                // 'guides/app/features/custom-charts/intro',
                'guides/app/features/custom-charts/walkthrough',
              ],
            },
            'guides/app/features/runs-table',
            'guides/app/features/tags',
            'guides/app/features/notes',
            'guides/app/features/teams',
            'guides/app/features/system-metrics',
            'guides/app/features/anon',
          ],
        },
        {
          type: 'category',
          label: 'Pages',
          link: {type: 'doc', id: 'guides/app/pages/intro'},
          items: [
            // 'guides/app/pages/intro',
            'guides/app/pages/gradient-panel',
            'guides/app/pages/project-page',
            'guides/app/pages/run-page',
            'guides/app/pages/workspaces',
          ],
        },
        {
          type: 'category',
          label: 'Settings',
          link: {type: 'doc', id: 'guides/app/settings-page/intro'},
          items: [
            // 'guides/app/settings-page/intro',
            'guides/app/settings-page/user-settings',
            'guides/app/settings-page/team-settings',
            'guides/app/settings-page/emails',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Private Hosting',
      link: {type: 'doc', id: 'guides/hosting/intro'},
      items: [
        // 'guides/hosting/intro',
        'guides/hosting/basic-setup',
        {
          type: 'category',
          label: 'Production Setup',
          link: {type: 'doc', id: 'guides/hosting/setup/intro'},
          items: [
            // 'guides/hosting/setup/intro',
            'guides/hosting/setup/dedicated-cloud',
            'guides/hosting/setup/private-cloud',
            'guides/hosting/setup/on-premise-baremetal',
            'guides/hosting/setup/configuration',
          ],
        },
        'guides/hosting/faq',
      ],
    },
    {
      type: 'category',
      label: 'Technical FAQ',
      link: {type: 'doc', id: 'guides/technical-faq/intro'},
      items: [
        // 'guides/hosting/technical-faq/intro',
        'guides/technical-faq/general',
        'guides/technical-faq/metrics-and-performance',
        'guides/technical-faq/setup',
        'guides/technical-faq/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      link: {type: 'doc', id: 'guides/integrations/intro'},
      items: [
        // 'guides/integrations/intro',
        'guides/integrations/add-wandb-to-any-library',
        'guides/integrations/other/catalyst',
        'guides/integrations/dagster',
        'guides/integrations/other/databricks',
        'guides/integrations/other/deepchecks',
        'guides/integrations/other/deepchem',
        'guides/integrations/other/docker',
        {
          type: 'category',
          label: 'Fastai',
          link: {type: 'doc', id: 'guides/integrations/fastai/README'},
          items: [
            // 'guides/integrations/fastai/README',
            'guides/integrations/fastai/v1',
          ],
        },
        'guides/integrations/huggingface',
        'guides/integrations/other/accelerate',
        'guides/integrations/other/hydra',
        'guides/integrations/keras',
        'guides/integrations/other/kubeflow-pipelines-kfp',
        'guides/integrations/lightgbm',
        'guides/integrations/other/metaflow',
        'guides/integrations/mmdetection',
        'guides/integrations/other/mmf',
        'guides/integrations/other/composer',
        'guides/integrations/other/openai',
        'guides/integrations/other/openai-gym',
        'guides/integrations/other/paddledetection',
        'guides/integrations/other/paddleocr',
        'guides/integrations/other/prodigy',
        'guides/integrations/pytorch',
        'guides/integrations/pytorch-geometric',
        'guides/integrations/other/ignite',
        'guides/integrations/lightning',
        'guides/integrations/other/ray-tune',
        'guides/integrations/other/sagemaker',
        'guides/integrations/scikit',
        'guides/integrations/other/simpletransformers',
        'guides/integrations/other/skorch',
        'guides/integrations/spacy',
        'guides/integrations/other/stable-baselines-3',
        'guides/integrations/tensorboard',
        'guides/integrations/tensorflow',
        'guides/integrations/other/w-and-b-for-julia',
        'guides/integrations/xgboost',
        'guides/integrations/yolov5',
        'guides/integrations/other/yolox',
        
      ],
    },
  ],
  ref: [
    {
      type: 'autogenerated',
      dirName: 'ref',
    },
  ],
};

module.exports = sidebars;
