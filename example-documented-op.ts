/**
 * @module Query Operations
 * @description Operations for querying and manipulating runs in W&B
 */

/**
 * Configuration dictionary type for W&B runs
 * @interface
 */
export interface ConfigDict {
  [key: string]: any;
  learning_rate?: number;
  batch_size?: number;
  epochs?: number;
  model_type?: string;
}

/**
 * Represents a W&B run with associated metadata and metrics
 * @interface
 */
export interface Run {
  id: string;
  name: string;
  project: string;
  entity: string;
  config: ConfigDict;
  summary: Record<string, any>;
  createdAt: Date;
  heartbeatAt: Date;
  state: 'running' | 'finished' | 'failed' | 'crashed';
}

/**
 * Gets the configuration object from a W&B run.
 * 
 * The configuration contains hyperparameters and settings used to initialize the run.
 * This is useful for comparing configurations across different experiments or
 * filtering runs based on specific parameter values.
 * 
 * @param run - The W&B run to extract configuration from
 * @returns The configuration dictionary containing all hyperparameters
 * 
 * @example
 * ```typescript
 * // Get configuration from a specific run
 * const config = runConfig(myRun);
 * console.log(config.learning_rate); // 0.001
 * console.log(config.batch_size); // 32
 * ```
 * 
 * @example
 * ```typescript
 * // Filter runs by configuration values
 * const runs = await project.runs();
 * const highLRRuns = runs.filter(run => {
 *   const config = runConfig(run);
 *   return config.learning_rate > 0.01;
 * });
 * ```
 * 
 * @category Chainable Operations
 * @since 1.0.0
 * @see {@link runSummary} - For accessing summary metrics
 * @see {@link runHistory} - For accessing time-series metrics
 */
export function runConfig(run: Run): ConfigDict {
  return run.config;
}

/**
 * Gets the summary metrics from a W&B run.
 * 
 * Summary metrics represent the final or best values logged during a run,
 * such as final accuracy, best validation loss, or total training time.
 * These are typically scalar values that summarize the run's performance.
 * 
 * @param run - The W&B run to extract summary from
 * @returns Dictionary of summary metrics
 * 
 * @example
 * ```typescript
 * // Get final metrics from a run
 * const summary = runSummary(myRun);
 * console.log(`Final accuracy: ${summary.accuracy}`);
 * console.log(`Best val loss: ${summary.best_val_loss}`);
 * ```
 * 
 * @example
 * ```typescript
 * // Compare summary metrics across runs
 * const runs = await project.runs();
 * const bestRun = runs.reduce((best, run) => {
 *   const summary = runSummary(run);
 *   const bestSummary = runSummary(best);
 *   return summary.accuracy > bestSummary.accuracy ? run : best;
 * });
 * ```
 * 
 * @category Chainable Operations
 * @since 1.0.0
 * @see {@link runConfig} - For accessing configuration
 * @see {@link runHistory} - For time-series metrics
 */
export function runSummary(run: Run): Record<string, any> {
  return run.summary;
}

/**
 * Gets the creation timestamp of a W&B run.
 * 
 * This timestamp indicates when the run was first initialized and can be used
 * for chronological sorting or filtering runs by date ranges.
 * 
 * @param run - The W&B run to get creation time from
 * @returns The creation date and time
 * 
 * @example
 * ```typescript
 * // Get runs from the last week
 * const oneWeekAgo = new Date();
 * oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
 * 
 * const recentRuns = runs.filter(run => {
 *   return runCreatedAt(run) > oneWeekAgo;
 * });
 * ```
 * 
 * @category Chainable Operations
 * @since 1.0.0
 */
export function runCreatedAt(run: Run): Date {
  return run.createdAt;
}

/**
 * Artifact version information
 * @interface
 */
export interface ArtifactVersion {
  id: string;
  name: string;
  alias: string;
  version: number;
  createdAt: Date;
  size: number;
  digest: string;
}

/**
 * Gets a specific artifact version logged by a run.
 * 
 * Artifacts in W&B are versioned files or directories that track model checkpoints,
 * datasets, or other outputs. This function retrieves a specific artifact version
 * that was logged during the run's execution.
 * 
 * @param run - The W&B run that logged the artifact
 * @param artifactVersionName - The artifact identifier in format "name:alias" (e.g., "model:latest" or "dataset:v2")
 * @returns The artifact version object, or undefined if not found
 * 
 * @example
 * ```typescript
 * // Get the latest model artifact from a run
 * const modelArtifact = runLoggedArtifactVersion(run, "model:latest");
 * if (modelArtifact) {
 *   console.log(`Model version: ${modelArtifact.version}`);
 *   console.log(`Model size: ${modelArtifact.size} bytes`);
 * }
 * ```
 * 
 * @example
 * ```typescript
 * // Check if a run logged a specific dataset version
 * const dataset = runLoggedArtifactVersion(run, "training-data:v3");
 * if (!dataset) {
 *   console.warn("Run did not use expected dataset version");
 * }
 * ```
 * 
 * @category Chainable Operations
 * @since 1.2.0
 * @see {@link runLoggedArtifactVersions} - Get all logged artifacts
 * @see {@link runUsedArtifactVersions} - Get input artifacts used by the run
 */
export function runLoggedArtifactVersion(
  run: Run, 
  artifactVersionName: string
): ArtifactVersion | undefined {
  // Implementation would fetch from W&B API
  throw new Error("Not implemented");
}
