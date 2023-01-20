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
      label: 'Runs',
      link: {type: 'doc', id:'guides/runs/intro'},
      items: [
        // 'guides/runs/intro',
        // 'guides/runs/create-run',
        'guides/track/advanced/grouping',
        'guides/track/advanced/resuming',
        'guides/track/advanced/alert',
      ]
    },
    {
      type: 'category',
      label: 'Track Experiments',
      link: {type: 'doc', id:'guides/track/intro'},
      items: [
        // 'guides/track/intro',
        'guides/track/launch',
        'guides/track/config',
        {
          type: 'category',
          label: 'Log Data From Experiments',
          link: {type: 'doc', id:'guides/track/log/intro'},
          items: [
            // 'guides/track/log/intro',
            'guides/track/log/plots',
            'guides/track/log/log-tables',
            'guides/track/log/log-summary',
            'guides/track/log/working-with-csv',
            'guides/track/log/distributed-training',
            'guides/track/log/customize-logging-axes',
            'guides/track/log/logging-faqs',
          ]
        },
        'guides/track/app',
        // 'guides/track/reproduce-experiments',
        'guides/track/jupyter',
        'guides/track/limits',
        'guides/track/public-api-guide',
        'guides/track/tracking-faq',
        'guides/track/advanced/save-restore',
        'guides/track/advanced/environment-variables',
      ]
    },
    {
      type: 'category',
      label: 'Artifacts',
      link: {type: 'doc', id:'guides/artifacts/intro'},
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
      ]
    },
    {
      type: 'category',
      label: 'Tune Hyperparameters',
      link: {type: 'doc', id:'guides/sweeps/intro'},
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
      ]
    },
    {
      type: 'category',
      label: 'Collaborative Reports',
      link: {type: 'doc', id:'guides/reports/intro'},
      items: [
        // 'guides/reports/intro',
        'guides/reports/create-a-report',
        'guides/reports/edit-a-report',
        'guides/reports/collaborate-on-reports',
        'guides/reports/clone-and-export-reports',
        'guides/reports/embed-reports',
        'guides/reports/cross-project-reports',
        'guides/reports/reports-faq',
      ]
    },
    'guides/launch/intro',
    {
      type: 'category',
      label: 'Data and model versioning',
      link: {type: 'doc', id:'guides/data-and-model-versioning/intro'},
      items: [
        // 'guides/data-and-model-versioning/intro',
        'guides/data-and-model-versioning/dataset-versioning',
        'guides/data-and-model-versioning/model-versioning',
      ]
    },
    {
      type: 'category',
      label: 'Model Management',
      link: {type: 'doc', id:'guides/models/intro'},
      items: [
        // 'guides/models/intro',
        'guides/models/model-management-concepts',
        'guides/models/walkthrough',
      ]
    },
    {
      type: 'category',
      label: 'Data Visualization',
      link: {type: 'doc', id:'guides/data-vis/intro'},
      items: [
        // 'guides/data-vis/intro',
        'guides/data-vis/tables-quickstart',
        'guides/data-vis/log-tables',
        'guides/data-vis/tables',
      ]
    },
    {
      type: 'category',
      label: 'W&B App UI',
      link: {type: 'doc', id:'guides/app/intro'},
      items: [
        // 'guides/app/intro',
        {
          type: 'category',
          label: 'Features',
          link: {type: 'doc', id:'guides/app/features/intro'},
          items: [
            // 'guides/app/features/intro',
            {
              type: 'category',
              label: 'Panels',
              link: {type: 'doc', id:'guides/app/features/panels/intro'},
              items: [
                // 'guides/app/features/panels/intro',
                {
                  type: 'category',
                  label: 'Line Plot',
                  link: {type: 'doc', id:'guides/app/features/panels/line-plot/intro'},
                  items: [
                    // 'guides/app/features/panels/line-plot/intro',
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
                  link: {type: 'doc', id:'guides/app/features/panels/weave/intro'},
                  items: [
                    // 'guides/app/features/panels/weave/intro',
                    'guides/app/features/panels/weave/embedding-projector',
                  ]
                },
              ]
            },
            {
              type: 'category',
              label: 'Custom Charts',
              link: {type: 'doc', id:'guides/app/features/custom-charts/intro'},
              items: [
                // 'guides/app/features/custom-charts/intro',
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
          link: {type: 'doc', id:'guides/app/pages/intro'},
          items: [
            // 'guides/app/pages/intro',
            'guides/app/pages/gradient-panel',
            'guides/app/pages/project-page',
            'guides/app/pages/run-page',
            'guides/app/pages/workspaces',
          ]
        },
        {
          type: 'category',
          label: 'Settings',
          link: {type: 'doc', id:'guides/app/settings-page/intro'},
          items: [
            // 'guides/app/settings-page/intro',
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
      link: {type: 'doc', id:'guides/hosting/intro'},
      items: [
        // 'guides/hosting/intro',
        'guides/hosting/basic-setup',
        {
          type: 'category',
          label: 'Production Setup',
          link: {type: 'doc', id:'guides/hosting/setup/intro'},
          items: [
            // 'guides/hosting/setup/intro',
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
      link: {type: 'doc', id:'guides/technical-faq/intro'},
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
      link: {type: 'doc', id:'guides/integrations/intro'},
      items: [
        // 'guides/integrations/intro',
        'guides/integrations/add-wandb-to-any-library',
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
          link: {type: 'doc', id:'guides/integrations/fastai/README'},
          items: [
            // 'guides/integrations/fastai/README',
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
  ref: [
        {
      type: 'autogenerated',
      dirName: 'ref',
    },
  ]
  // ref: [
  //   'ref/README',
  //   {
  //     type: 'category',
  //     label: 'Python Library',
  //     items: [
  //       'ref/python/README',
  //       'ref/python/artifact',
  //       'ref/python/run',
  //       'ref/python/agent',
  //       'ref/python/controller',
  //       'ref/python/finish',
  //       'ref/python/init',
  //       'ref/python/log',
  //       'ref/python/save',
  //       'ref/python/sweep',
  //       'ref/python/watch',
  //       {
  //         type: 'category',
  //         label: 'Import and Export API',
  //         items: [
  //           'ref/python/public-api/README',
  //           'ref/python/public-api/api',
  //           'ref/python/public-api/artifact',
  //           'ref/python/public-api/file',
  //           'ref/python/public-api/files',
  //           'ref/python/public-api/project',
  //           'ref/python/public-api/projects',
  //           'ref/python/public-api/run',
  //           'ref/python/public-api/runs',
  //           'ref/python/public-api/sweep',
  //         ]
  //       },
  //       {
  //         type: 'category',
  //         label: 'Integration API',
  //         items: [
  //           'ref/python/integrations/keras/README',
  //           'ref/python/integrations/keras/wandbcallback',
  //           'ref/python/integrations/keras/wandbevalcallback',
  //           'ref/python/integrations/keras/wandbmetricslogger',
  //           'ref/python/integrations/keras/wandbmodelcheckpoint',
  //         ]
  //       },
  //       {
  //         type: 'category',
  //         label: 'Visualizatin Data Types',
  //         items: [
  //           'ref/python/data-types/README',
  //           'ref/python/data-types/audio',
  //           'ref/python/data-types/boundingboxes2d',
  //           'ref/python/data-types/graph',
  //           'ref/python/data-types/histogram',
  //           'ref/python/data-types/html',
  //           'ref/python/data-types/image',
  //           'ref/python/data-types/imagemask',
  //           'ref/python/data-types/molecule',
  //           'ref/python/data-types/object3d',
  //           'ref/python/data-types/plotly',
  //           'ref/python/data-types/table',
  //           'ref/python/data-types/video',
  //         ]
  //       },
  //     ]
  //   },
  //   {
  //     type: 'category',
  //     label: 'Command Line Interface',
  //     items: [
  //       'ref/cli/README',
  //       {
  //         type: 'category',
  //         label: 'wandb artifact',
  //         items: [
  //           'ref/cli/wandb-artifact/README',
  //           'ref/cli/wandb-artifact/wandb-artifact-cache/README',
  //           'ref/cli/wandb-artifact/wandb-artifact-cache/wandb-artifact-cache-cleanup',
  //           'ref/cli/wandb-artifact/wandb-artifact-get',
  //           'ref/cli/wandb-artifact/wandb-artifact-ls',
  //           'ref/cli/wandb-artifact/wandb-artifact-put',
  //         ]
  //       },
  //       'ref/cli/wandb-agent',
  //       'ref/cli/wandb-controller',
  //       'ref/cli/wandb-disabled',
  //       'ref/cli/wandb-docker-run',
  //       'ref/cli/wandb-docker',
  //       'ref/cli/wandb-enabled',
  //       'ref/cli/wandb-init',
  //       'ref/cli/wandb-launch-agent',
  //       'ref/cli/wandb-launch',
  //       'ref/cli/wandb-login',
  //       'ref/cli/wandb-offline',
  //       'ref/cli/wandb-online',
  //       'ref/cli/wandb-pull',
  //       'ref/cli/wandb-restore',
  //       'ref/cli/wandb-scheduler',
  //       {
  //         type: 'category',
  //         label: 'wandb server',
  //         items: [
  //           'ref/cli/wandb-server/README',
  //           'ref/cli/wandb-server/wandb-server-start',
  //           'ref/cli/wandb-server/wandb-server-stop',
  //         ]
  //       },
  //       'ref/cli/wandb-status',
  //       'ref/cli/wandb-sweep',
  //       'ref/cli/wandb-sync',
  //       'ref/cli/wandb-verify',
  //     ]
  //   },
  //   {
  //     type: 'category',
  //     label: 'Java Library',
  //     items: [
  //       'ref/java/README',
  //       'ref/java/wandbrun',
  //       'ref/java/wandbrun-builder',
  //     ]
  //   },
  //   {
  //     type: 'category',
  //     label: 'Weave',
  //     items: [
  //       'ref/weave/README',
  //       'ref/weave/artifact-type',
  //       'ref/weave/artifact-version',
  //       'ref/weave/artifact',
  //       'ref/weave/audio-file',
  //       'ref/weave/bokeh-file',
  //       'ref/weave/boolean',
  //       'ref/weave/entity',
  //       'ref/weave/file',
  //       'ref/weave/float',
  //       'ref/weave/html-file',
  //       'ref/weave/image-file',
  //       'ref/weave/int',
  //       'ref/weave/joined-table',
  //       'ref/weave/molecule-file',
  //       'ref/weave/number',
  //       'ref/weave/object-3-d-file',
  //       'ref/weave/partitioned-table',
  //       'ref/weave/project',
  //       'ref/weave/pytorch-model-file',
  //       'ref/weave/run',
  //       'ref/weave/string',
  //       'ref/weave/table',
  //       'ref/weave/user',
  //       'ref/weave/video-file',
  //     ]
  //   },
  // ]
};

module.exports = sidebars;
