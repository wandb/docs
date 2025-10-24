#!/usr/bin/env node

/**
 * MDX Validation Script
 * 
 * This script validates MDX files by attempting to compile them
 * using the @mdx-js/mdx compiler. It catches parsing errors that
 * might otherwise only appear in the Mintlify dev server logs.
 */

const fs = require('fs').promises;
const path = require('path');
const { compile } = require('@mdx-js/mdx');
const { glob } = require('glob');

// ANSI color codes for terminal output
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

/**
 * Validate a single MDX file
 * @param {string} filePath - Path to the MDX file
 * @returns {Promise<{success: boolean, error?: string}>}
 */
async function validateMDXFile(filePath) {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    
    // Attempt to compile the MDX content
    await compile(content, {
      // Use development mode for better error messages
      development: true,
      // Format as 'mdx' to handle MDX-specific syntax
      format: 'mdx'
    });
    
    return { success: true };
  } catch (error) {
    // Extract useful error information
    let errorMessage = error.message || 'Unknown error';
    
    // Try to extract line and column information if available
    if (error.line && error.column) {
      errorMessage = `Line ${error.line}, Column ${error.column}: ${errorMessage}`;
    } else if (error.position) {
      const { start } = error.position;
      if (start) {
        errorMessage = `Line ${start.line}, Column ${start.column}: ${errorMessage}`;
      }
    }
    
    return { 
      success: false, 
      error: errorMessage
    };
  }
}

/**
 * Get list of MDX files to validate
 * @param {string[]} patterns - Glob patterns or file paths
 * @returns {Promise<string[]>}
 */
async function getMDXFiles(patterns) {
  const files = new Set();
  
  for (const pattern of patterns) {
    // Check if it's a specific file
    if (pattern.endsWith('.mdx')) {
      try {
        await fs.access(pattern);
        files.add(pattern);
      } catch {
        // If it's not a file, treat it as a glob pattern
        const matches = await glob(pattern, {
          ignore: ['node_modules/**', '.next/**', 'dist/**', 'build/**']
        });
        matches.forEach(file => files.add(file));
      }
    } else {
      // Treat as a glob pattern
      const globPattern = pattern.endsWith('/') ? `${pattern}**/*.mdx` : `${pattern}/**/*.mdx`;
      const matches = await glob(globPattern, {
        ignore: ['node_modules/**', '.next/**', 'dist/**', 'build/**']
      });
      matches.forEach(file => files.add(file));
    }
  }
  
  return Array.from(files);
}

/**
 * Main validation function
 */
async function main() {
  const args = process.argv.slice(2);
  
  // Parse command line arguments
  let patterns = [];
  let isCI = false;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--ci') {
      isCI = true;
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log(`
${colors.bold}MDX Validation Script${colors.reset}

Usage: node validate-mdx.js [options] [files/patterns...]

Options:
  --ci          Run in CI mode (exit with error code on failures)
  --help, -h    Show this help message

Examples:
  node validate-mdx.js                           # Validate all MDX files
  node validate-mdx.js docs/                     # Validate MDX files in docs/
  node validate-mdx.js file.mdx                  # Validate specific file
  node validate-mdx.js "**/*.mdx"                # Validate all MDX files (glob)
  node validate-mdx.js --ci docs/**/*.mdx        # CI mode for docs directory
      `);
      process.exit(0);
    } else {
      patterns.push(args[i]);
    }
  }
  
  // Default to all MDX files if no patterns provided
  if (patterns.length === 0) {
    patterns = ['**/*.mdx'];
  }
  
  // Get list of files to validate
  const files = await getMDXFiles(patterns);
  
  if (files.length === 0) {
    console.log(`${colors.yellow}No MDX files found to validate${colors.reset}`);
    process.exit(0);
  }
  
  console.log(`${colors.blue}Validating ${files.length} MDX file(s)...${colors.reset}\n`);
  
  let totalErrors = 0;
  const errors = [];
  
  // Validate each file
  for (const file of files) {
    const result = await validateMDXFile(file);
    
    if (result.success) {
      if (!isCI) {
        console.log(`${colors.green}✓${colors.reset} ${file}`);
      }
    } else {
      totalErrors++;
      const errorInfo = {
        file,
        error: result.error
      };
      errors.push(errorInfo);
      
      console.log(`${colors.red}✗${colors.reset} ${file}`);
      console.log(`  ${colors.red}Error: ${result.error}${colors.reset}`);
    }
  }
  
  // Summary
  console.log('\n' + '='.repeat(60));
  
  if (totalErrors === 0) {
    console.log(`${colors.green}${colors.bold}✓ All MDX files validated successfully!${colors.reset}`);
    process.exit(0);
  } else {
    console.log(`${colors.red}${colors.bold}✗ Found ${totalErrors} file(s) with parsing errors${colors.reset}`);
    
    if (isCI) {
      console.log('\n' + colors.red + 'MDX validation failed. Please fix the parsing errors before merging.' + colors.reset);
      process.exit(1);
    } else {
      console.log('\n' + colors.yellow + 'Please fix the parsing errors listed above.' + colors.reset);
      process.exit(1);
    }
  }
}

// Run the script
main().catch(error => {
  console.error(`${colors.red}Unexpected error: ${error.message}${colors.reset}`);
  process.exit(1);
});
