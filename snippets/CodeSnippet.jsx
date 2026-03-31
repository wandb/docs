/**
 * CodeSnippet component for including Python code examples by filename
 * Usage: <CodeSnippet file="artifact_create.py" />
 * 
 * This component consolidates all code example imports in one place,
 * allowing cheat sheet pages to reference snippets by filename without
 * needing individual import statements.
 * 
 * Note: Python files are kept as source of truth, with MDX wrappers
 * generated for compatibility with Mintlify's import system.
 * 
 * AUTO-GENERATED: Do not edit manually. Run sync_code_examples.sh to regenerate.
 */

import React from 'react';

// Import all MDX-wrapped code examples
import ArtifactAddAlias from '/snippets/en/_includes/code-examples/artifact_add_alias.mdx';
import ArtifactAddAliasExisting from '/snippets/en/_includes/code-examples/artifact_add_alias_existing.mdx';
import ArtifactAddTag from '/snippets/en/_includes/code-examples/artifact_add_tag.mdx';
import ArtifactAddTagExisting from '/snippets/en/_includes/code-examples/artifact_add_tag_existing.mdx';
import ArtifactCreate from '/snippets/en/_includes/code-examples/artifact_create.mdx';
import ArtifactCreateTrackExternal from '/snippets/en/_includes/code-examples/artifact_create_track_external.mdx';
import ArtifactDelete from '/snippets/en/_includes/code-examples/artifact_delete.mdx';
import ArtifactDeleteCollection from '/snippets/en/_includes/code-examples/artifact_delete_collection.mdx';
import ArtifactDownloadPartial from '/snippets/en/_includes/code-examples/artifact_download_partial.mdx';
import ArtifactTtl from '/snippets/en/_includes/code-examples/artifact_ttl.mdx';
import ArtifactTtlExisting from '/snippets/en/_includes/code-examples/artifact_ttl_existing.mdx';
import ArtifactUpdate from '/snippets/en/_includes/code-examples/artifact_update.mdx';
import ArtifactUpdateExisting from '/snippets/en/_includes/code-examples/artifact_update_existing.mdx';
import ArtifactsDownload from '/snippets/en/_includes/code-examples/artifacts_download.mdx';
import ExperimentsCreate from '/snippets/en/_includes/code-examples/experiments_create.mdx';
import LogCsvFiles from '/snippets/en/_includes/code-examples/log_csv_files.mdx';
import LogCustomSummaryMetric from '/snippets/en/_includes/code-examples/log_custom_summary_metric.mdx';
import LogCustomSummatyMetric from '/snippets/en/_includes/code-examples/log_custom_summaty_metric.mdx';
import LogExistingArtifactLinkCollection from '/snippets/en/_includes/code-examples/log_existing_artifact_link_collection.mdx';
import LogHistogramPlot from '/snippets/en/_includes/code-examples/log_histogram_plot.mdx';
import LogHyperparameter from '/snippets/en/_includes/code-examples/log_hyperparameter.mdx';
import LogLinePlot from '/snippets/en/_includes/code-examples/log_line_plot.mdx';
import LogMetric from '/snippets/en/_includes/code-examples/log_metric.mdx';
import LogScatterPlot from '/snippets/en/_includes/code-examples/log_scatter_plot.mdx';
import LogSummaryMetric from '/snippets/en/_includes/code-examples/log_summary_metric.mdx';
import LogTable from '/snippets/en/_includes/code-examples/log_table.mdx';
import RegistryAddAnnotation from '/snippets/en/_includes/code-examples/registry_add_annotation.mdx';
import RegistryCollectionCreate from '/snippets/en/_includes/code-examples/registry_collection_create.mdx';
import RegistryCollectionTagsAdd from '/snippets/en/_includes/code-examples/registry_collection_tags_add.mdx';
import RegistryCollectionTagsRemove from '/snippets/en/_includes/code-examples/registry_collection_tags_remove.mdx';
import RegistryCreate from '/snippets/en/_includes/code-examples/registry_create.mdx';
import RegistryDelete from '/snippets/en/_includes/code-examples/registry_delete.mdx';
import RegistryLinkArtifactExisting from '/snippets/en/_includes/code-examples/registry_link_artifact_existing.mdx';
import RegistryUseLinkedArtifact from '/snippets/en/_includes/code-examples/registry_use_linked_artifact.mdx';
import ResumeRun from '/snippets/en/_includes/code-examples/resume_run.mdx';
import RewindRun from '/snippets/en/_includes/code-examples/rewind_run.mdx';
import RunFork from '/snippets/en/_includes/code-examples/run_fork.mdx';
import RunInit from '/snippets/en/_includes/code-examples/run_init.mdx';
import RunsAddTags from '/snippets/en/_includes/code-examples/runs_add_tags.mdx';
import RunsGroup from '/snippets/en/_includes/code-examples/runs_group.mdx';
import RunsOrganizeByType from '/snippets/en/_includes/code-examples/runs_organize_by_type.mdx';
import RunsUpdateTag from '/snippets/en/_includes/code-examples/runs_update_tag.mdx';
import RunsUpdateTagPublicApi from '/snippets/en/_includes/code-examples/runs_update_tag_public_api.mdx';
import SweepConfig from '/snippets/en/_includes/code-examples/sweep_config.mdx';
import SweepCreate from '/snippets/en/_includes/code-examples/sweep_create.mdx';
import SweepInitialize from '/snippets/en/_includes/code-examples/sweep_initialize.mdx';
import SweepStart from '/snippets/en/_includes/code-examples/sweep_start.mdx';

// Map filenames to imported content
const snippets = {
  'artifact_add_alias.py': ArtifactAddAlias,
  'artifact_add_alias_existing.py': ArtifactAddAliasExisting,
  'artifact_add_tag.py': ArtifactAddTag,
  'artifact_add_tag_existing.py': ArtifactAddTagExisting,
  'artifact_create.py': ArtifactCreate,
  'artifact_create_track_external.py': ArtifactCreateTrackExternal,
  'artifact_delete.py': ArtifactDelete,
  'artifact_delete_collection.py': ArtifactDeleteCollection,
  'artifact_download_partial.py': ArtifactDownloadPartial,
  'artifact_ttl.py': ArtifactTtl,
  'artifact_ttl_existing.py': ArtifactTtlExisting,
  'artifact_update.py': ArtifactUpdate,
  'artifact_update_existing.py': ArtifactUpdateExisting,
  'artifacts_download.py': ArtifactsDownload,
  'experiments_create.py': ExperimentsCreate,
  'log_csv_files.py': LogCsvFiles,
  'log_custom_summary_metric.py': LogCustomSummaryMetric,
  'log_custom_summaty_metric.py': LogCustomSummatyMetric,
  'log_existing_artifact_link_collection.py': LogExistingArtifactLinkCollection,
  'log_histogram_plot.py': LogHistogramPlot,
  'log_hyperparameter.py': LogHyperparameter,
  'log_line_plot.py': LogLinePlot,
  'log_metric.py': LogMetric,
  'log_scatter_plot.py': LogScatterPlot,
  'log_summary_metric.py': LogSummaryMetric,
  'log_table.py': LogTable,
  'registry_add_annotation.py': RegistryAddAnnotation,
  'registry_collection_create.py': RegistryCollectionCreate,
  'registry_collection_tags_add.py': RegistryCollectionTagsAdd,
  'registry_collection_tags_remove.py': RegistryCollectionTagsRemove,
  'registry_create.py': RegistryCreate,
  'registry_delete.py': RegistryDelete,
  'registry_link_artifact_existing.py': RegistryLinkArtifactExisting,
  'registry_use_linked_artifact.py': RegistryUseLinkedArtifact,
  'resume_run.py': ResumeRun,
  'rewind_run.py': RewindRun,
  'run_fork.py': RunFork,
  'run_init.py': RunInit,
  'runs_add_tags.py': RunsAddTags,
  'runs_group.py': RunsGroup,
  'runs_organize_by_type.py': RunsOrganizeByType,
  'runs_update_tag.py': RunsUpdateTag,
  'runs_update_tag_public_api.py': RunsUpdateTagPublicApi,
  'sweep_config.py': SweepConfig,
  'sweep_create.py': SweepCreate,
  'sweep_initialize.py': SweepInitialize,
  'sweep_start.py': SweepStart,
};

export const CodeSnippet = ({ file }) => {
  const Component = snippets[file];
  
  if (!Component) {
    return (
      <div style={{ padding: '1rem', background: '#fee', border: '1px solid #fcc', borderRadius: '4px' }}>
        <p style={{ margin: 0, color: '#c00' }}>Code snippet not found: {file}</p>
      </div>
    );
  }
  
  return <Component />;
};

export default CodeSnippet;
