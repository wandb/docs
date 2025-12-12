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
import ArtifactAddAlias from './en/_includes/code-examples/artifact_add_alias.mdx';
import ArtifactAddAliasExisting from './en/_includes/code-examples/artifact_add_alias_existing.mdx';
import ArtifactAddTag from './en/_includes/code-examples/artifact_add_tag.mdx';
import ArtifactAddTagExisting from './en/_includes/code-examples/artifact_add_tag_existing.mdx';
import ArtifactCreate from './en/_includes/code-examples/artifact_create.mdx';
import ArtifactCreateTrackExternal from './en/_includes/code-examples/artifact_create_track_external.mdx';
import ArtifactDelete from './en/_includes/code-examples/artifact_delete.mdx';
import ArtifactDeleteCollection from './en/_includes/code-examples/artifact_delete_collection.mdx';
import ArtifactDownloadPartial from './en/_includes/code-examples/artifact_download_partial.mdx';
import ArtifactTtl from './en/_includes/code-examples/artifact_ttl.mdx';
import ArtifactTtlExisting from './en/_includes/code-examples/artifact_ttl_existing.mdx';
import ArtifactUpdate from './en/_includes/code-examples/artifact_update.mdx';
import ArtifactUpdateExisting from './en/_includes/code-examples/artifact_update_existing.mdx';
import ArtifactsDownload from './en/_includes/code-examples/artifacts_download.mdx';
import ExperimentsCreate from './en/_includes/code-examples/experiments_create.mdx';
import LogCustomSummaryMetric from './en/_includes/code-examples/log_custom_summary_metric.mdx';
import LogExistingArtifactLinkCollection from './en/_includes/code-examples/log_existing_artifact_link_collection.mdx';
import LogHyperparameter from './en/_includes/code-examples/log_hyperparameter.mdx';
import LogMetric from './en/_includes/code-examples/log_metric.mdx';
import LogTable from './en/_includes/code-examples/log_table.mdx';
import RegistryAddAnnotation from './en/_includes/code-examples/registry_add_annotation.mdx';
import RegistryCollectionCreate from './en/_includes/code-examples/registry_collection_create.mdx';
import RegistryCollectionTagsAdd from './en/_includes/code-examples/registry_collection_tags_add.mdx';
import RegistryCollectionTagsRemove from './en/_includes/code-examples/registry_collection_tags_remove.mdx';
import RegistryCreate from './en/_includes/code-examples/registry_create.mdx';
import RegistryDelete from './en/_includes/code-examples/registry_delete.mdx';
import RegistryLinkArtifactExisting from './en/_includes/code-examples/registry_link_artifact_existing.mdx';
import RegistryUseLinkedArtifact from './en/_includes/code-examples/registry_use_linked_artifact.mdx';
import RunFork from './en/_includes/code-examples/run_fork.mdx';
import RunInit from './en/_includes/code-examples/run_init.mdx';

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
  'log_custom_summary_metric.py': LogCustomSummaryMetric,
  'log_existing_artifact_link_collection.py': LogExistingArtifactLinkCollection,
  'log_hyperparameter.py': LogHyperparameter,
  'log_metric.py': LogMetric,
  'log_table.py': LogTable,
  'registry_add_annotation.py': RegistryAddAnnotation,
  'registry_collection_create.py': RegistryCollectionCreate,
  'registry_collection_tags_add.py': RegistryCollectionTagsAdd,
  'registry_collection_tags_remove.py': RegistryCollectionTagsRemove,
  'registry_create.py': RegistryCreate,
  'registry_delete.py': RegistryDelete,
  'registry_link_artifact_existing.py': RegistryLinkArtifactExisting,
  'registry_use_linked_artifact.py': RegistryUseLinkedArtifact,
  'run_fork.py': RunFork,
  'run_init.py': RunInit,
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
