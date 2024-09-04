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
  default: [
    'guides/intro',
    'quickstart',
    {
      type: 'category',
      label: 'W&B Models',
      link: {type: 'doc', id: 'guides/models'},
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Experiments',
          link: {type: 'doc', id: 'guides/track/intro'},
          items: [
            // 'guides/track/intro',
            'guides/track/launch',
            'guides/track/config',
            'guides/app/pages/project-page',
            'guides/app/pages/workspaces',
            {
              type: 'category',
              label: 'What are runs?',
              link: {type: 'doc', id: 'guides/runs/intro'},
              items: [
                // 'guides/runs/create-run',
                'guides/app/pages/run-page',
                'guides/runs/grouping',
                'guides/runs/resuming',
                'guides/runs/rewind',
                'guides/runs/forking',
                'guides/runs/alert',
                'guides/runs/manage-runs',
              ],
            },
            {
              type: 'category',
              label: 'Log Objects and Media',
              link: {type: 'doc', id: 'guides/track/log/intro'},
              items: [
                'guides/track/log/log-models',
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
            'guides/track/environment-variables',
          ],
        },
        { 
          type: 'category',
          label: 'Sweeps',
          link: {type: 'doc', id: 'guides/sweeps/intro'},
          items: [
            'guides/sweeps/walkthrough',
            'guides/sweeps/add-w-and-b-to-your-code',
            {
              type: 'category',
              label: 'Define a sweep configuration',
              items: [
                'guides/sweeps/define-sweep-configuration',
                'guides/sweeps/sweep-config-keys',
              ],
            },
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
          label: 'Registry',
          link: {type: 'doc', id: 'guides/registry/intro'},
          items: [
            'guides/registry/registry_types',
            'guides/registry/create_registry',
            'guides/registry/configure_registry',
            'guides/registry/create_collection',
            'guides/registry/link_version',
            'guides/registry/download_use_artifact',
            'guides/registry/model_registry_eol',
            {
              type: 'category',
              label: 'Model Registry',
              link: {type: 'doc', id: 'guides/model_registry/intro'},
              items: [
                'guides/model_registry/walkthrough',
                'guides/model_registry/model-management-concepts',
                'guides/model_registry/log-model-to-experiment',
                'guides/model_registry/create-registered-model',
                'guides/model_registry/link-model-version',
                'guides/model_registry/organize-models',
                'guides/model_registry/model-lineage',
                'guides/model_registry/consume-models',
                'guides/model_registry/create-model-cards',
                // 'guides/model_registry/delete-models',
                'guides/model_registry/notifications',
                'guides/model_registry/access_controls',
              ],
            },
          ],
        },       
        {
          type: 'category',
          label: 'Automations',
          items: [
            'guides/model_registry/model-registry-automations',
            'guides/artifacts/project-scoped-automations',
          ]
        },
        {
          type: 'category',
          label: 'Launch',
          link: {
            type: 'doc',
            id: 'guides/launch/intro',
          },
          items: [
            'guides/launch/walkthrough',
            'guides/launch/launch-terminology',
            {
              type: 'category',
              label: 'Set up Launch',
              link: {
                type: 'doc',
                id: 'guides/launch/setup-launch',
              },
              items: [
                'guides/launch/setup-launch-docker',
                'guides/launch/setup-launch-sagemaker',
                'guides/launch/setup-launch-kubernetes',
                'guides/launch/setup-vertex',
                'guides/launch/setup-agent-advanced',
                'guides/launch/setup-queue-advanced',
              ],
            },
            {
              type: 'category',
              label: 'Create and deploy jobs',
              items: [
                'guides/launch/create-launch-job',
                'guides/launch/add-job-to-queue',
                'guides/launch/job-inputs',
                'guides/launch/launch-view-jobs',
                'guides/launch/launch-queue-observability',
              ],
            },
            'guides/launch/sweeps-on-launch',
            'guides/launch/launch-faqs',
          ],
        },
        {
          type: 'category',
          label: 'W&B App UI Reference',
          link: {type: 'doc', id: 'guides/app/intro'},
          items: [
            {
              type: 'category',
              label: 'Features',
              link: {type: 'doc', id: 'guides/app/features/intro'},
              items: [
                {
                  type: 'category',
                  label: 'Panels',
                  link: {type: 'doc', id: 'guides/app/features/panels/intro'},
                  items: [
                    {
                      type: 'category',
                      label: 'Line Plot',
                      link: {
                        type: 'doc',
                        id: 'guides/app/features/panels/line-plot/intro',
                      },
                      items: [
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
                      label: 'Query panels',
                      link: {
                        type: 'doc',
                        id: 'guides/app/features/panels/query-panel/intro',
                      },
                      items: [
                        'guides/app/features/panels/query-panel/embedding-projector',
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
              ],
            },
            {
              type: 'category',
              label: 'Settings',
              link: {type: 'doc', id: 'guides/app/settings-page/intro'},
              items: [
                'guides/app/settings-page/user-settings',
                'guides/app/settings-page/team-settings',
                'guides/app/settings-page/emails',
                'guides/app/features/teams',
                'guides/app/features/organizations',
                'guides/app/features/storage',
                'guides/app/features/system-metrics',
                'guides/app/features/anon',
              ],
            },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'W&B Weave',
      link: { type: 'doc', id: 'guides/weave_platform'},
      items: [],
    },
    {
      type: 'category',
      label: 'W&B Core',
      link: {type: 'doc', id: 'guides/core'},
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Artifacts',
          link: {type: 'doc', id: 'guides/artifacts/intro'},
          items: [
            'guides/artifacts/artifacts-walkthrough',
            'guides/artifacts/construct-an-artifact',
            'guides/artifacts/download-and-use-an-artifact',
            'guides/artifacts/update-an-artifact',
            'guides/artifacts/create-a-custom-alias',
            'guides/artifacts/create-a-new-artifact-version',
            'guides/artifacts/track-external-files',
            {
              type: 'category',
              label: 'Manage data',
              items: [
                'guides/artifacts/delete-artifacts',
                'guides/artifacts/storage',
                'guides/artifacts/ttl',
              ],
            },
            'guides/artifacts/explore-and-traverse-an-artifact-graph',
            'guides/artifacts/data-privacy-and-compliance',
            'guides/artifacts/artifacts-faqs',
            // 'guides/artifacts/examples',
          ],
        },
        {
          type: 'category',
          label: 'Tables',
          link: {type: 'doc', id: 'guides/tables/intro'},
          items: [
            'guides/tables/tables-walkthrough',
            'guides/tables/visualize-tables',
            'guides/tables/tables-gallery',
            'guides/tables/tables-download',
          ],
        },
        {
          type: 'category',
          label: 'Reports',
          link: {type: 'doc', id: 'guides/reports/intro'},
          items: [
            'guides/reports/create-a-report',
            'guides/reports/edit-a-report',
            'guides/reports/collaborate-on-reports',
            'guides/reports/clone-and-export-reports',
            'guides/reports/embed-reports',
            'guides/reports/cross-project-reports',
            'guides/reports/reports-gallery',
            'guides/reports/reports-faq',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'W&B Platform',
      link: { type: 'doc', id: 'guides/hosting/intro' },
      items: [
        {
          type: 'category',
          label: 'Deployment options',
          items: [
            'guides/hosting/hosting-options/saas_cloud',
            {
              type: 'category',
              label: 'Dedicated Cloud',
              link: { type: 'doc', id: 'guides/hosting/hosting-options/dedicated_cloud' },
              items: [
                'guides/hosting/hosting-options/dedicated_regions',
                'guides/hosting/export-data-from-dedicated-cloud',
              ],
            },
            {
              type: 'category',
              label: 'Self Managed',
              link: { type: 'doc', id: 'guides/hosting/hosting-options/self-managed' },
              items: [
                'guides/hosting/self-managed/basic-setup',
                'guides/hosting/operator',  
                {
                  type: 'category',
                  label: 'Install on public cloud',
                  items: [
                    'guides/hosting/self-managed/aws-tf',
                    'guides/hosting/self-managed/gcp-tf',
                    'guides/hosting/self-managed/azure-tf',
                  ],
                },
                'guides/hosting/self-managed/bare-metal',
                'guides/hosting/server-upgrade-process',
              ],
            },
          ],
        },        
        {
          type: 'category',
          label: 'Identity and access management (IAM)',
          link: { type: 'doc', id: 'guides/hosting/iam/org_team_struct'},
          items: [
            {
              type: 'category',
              label: 'Authentication',
              items: [
                'guides/hosting/iam/sso',
                'guides/hosting/iam/ldap',
                'guides/hosting/iam/identity_federation'
              ],
            },
            {
              type: 'category',
              label: 'Access management',
              items: [
                'guides/hosting/iam/manage-users',
                'guides/hosting/iam/restricted-projects',
              ],
            },
            'guides/hosting/iam/automate_iam',
            'guides/hosting/iam/scim',
            'guides/hosting/iam/advanced_env_vars',               
          ],
        },
        {
          type: 'category',
          label: 'Data Security',
          items: [
            'guides/hosting/data-security/secure-storage-connector',
            'guides/hosting/data-security/presigned-urls',
            'guides/hosting/data-security/ip-allowlisting',
            'guides/hosting/data-security/private-connectivity',
            'guides/hosting/data-security/data-encryption',
          ],
        },
        'guides/hosting/privacy-settings',               
        {
          type: 'category',
          label: 'Monitoring and Usage',
          items: [
            'guides/hosting/monitoring-usage/audit-logging',
            'guides/hosting/monitoring-usage/prometheus-logging',
            'guides/hosting/monitoring-usage/slack-alerts',
            'guides/hosting/monitoring-usage/org_dashboard',
          ],
        },
        'guides/hosting/smtp',
        'guides/hosting/env-vars',
        'guides/hosting/server-release-process',
        // 'guides/hosting/debug',
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      link: {type: 'doc', id: 'guides/integrations/intro'},
      items: [
        'guides/integrations/add-wandb-to-any-library',
        'guides/integrations/other/azure-openai-fine-tuning',
        'guides/integrations/other/catalyst',
        'guides/integrations/dagster',
        'guides/integrations/other/databricks',
        'guides/integrations/other/deepchecks',
        'guides/integrations/other/deepchem',
        'guides/integrations/other/docker',
        'guides/integrations/other/farama-gymnasium',
        {
          type: 'category',
          label: 'Fastai',
          link: {type: 'doc', id: 'guides/integrations/fastai/README'},
          items: [
            'guides/integrations/fastai/v1',
          ],
        },
        'guides/integrations/huggingface',
        'guides/integrations/diffusers',
        'guides/integrations/autotrain',
        'guides/integrations/other/accelerate',
        'guides/integrations/other/hydra',
        'guides/integrations/keras',
        'guides/integrations/other/kubeflow-pipelines-kfp',
        'guides/integrations/langchain',
        'guides/integrations/lightgbm',
        'guides/integrations/other/metaflow',
        'guides/integrations/mmengine',
        'guides/integrations/other/mmf',
        'guides/integrations/other/composer',
        'guides/integrations/nim',
        'guides/integrations/other/openai-api',
        'guides/integrations/other/openai-fine-tuning',
        'guides/integrations/other/openai-gym',
        'guides/integrations/other/paddledetection',
        'guides/integrations/other/paddleocr',
        'guides/integrations/other/prodigy',
        'guides/integrations/pytorch',
        'guides/integrations/pytorch-geometric',
        'guides/integrations/torchtune',
        'guides/integrations/other/ignite',
        'guides/integrations/lightning',
        {
          type: 'category',
          label: 'Prompts',
          link: {
            type: 'doc',
            id: 'guides/integrations/prompts/intro'
          },
          items: [
            'guides/integrations/prompts/quickstart', 
            'guides/integrations/prompts/openai'
          ],
        },
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
        'guides/integrations/ultralytics',
        'guides/integrations/other/yolox',
      ],
    },
    {
      type: 'category',
      label: 'Technical FAQ',
      link: {type: 'doc', id: 'guides/technical-faq/intro'},
      items: [
        // 'guides/hosting/technical-faq/intro',
        'guides/technical-faq/general',
        'guides/technical-faq/admin',
        'guides/technical-faq/metrics-and-performance',
        'guides/technical-faq/setup',
        'guides/technical-faq/troubleshooting',
      ],
    },
  ],
  ref: [
    {
      type: 'autogenerated',
      dirName: 'ref',
    },
  ],
  tutorials: [
    {
      type: 'doc',
      id: 'tutorials/intro_to_tutorials', // document ID
      label: 'W&B Tutorials', // sidebar label
    },
    'tutorials/experiments',
    'tutorials/tables',
    'tutorials/sweeps',
    'tutorials/artifacts',
    'tutorials/workspaces',
    {
      type: 'category',
      label: 'Integration Tutorials',
      items: [
        'tutorials/pytorch',
        'tutorials/lightning',
        'tutorials/huggingface',
        'tutorials/tensorflow',
        'tutorials/tensorflow_sweeps',
        'tutorials/keras',
        'tutorials/keras_tables',
        'tutorials/keras_models',
        // 'tutorials/xgboost',  # broken
        'tutorials/xgboost_sweeps',
        // 'tutorials/lightgbm', # broken
        'tutorials/monai_3d_segmentation',
      ],
    },
    {
      type: 'category',
      label: 'Launch Tutorials',
      items: ['tutorials/volcano', 'tutorials/minikube_gpu'],
    },
  ],
};

module.exports = sidebars;
