const fs = require('fs');
const path = require('path');

// Function to get all markdown files in the directory and subdirectories
function getMarkdownFiles(dir, fileList = []) {
    const files = fs.readdirSync(dir);

    files.forEach(file => {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory()) {
            fileList = getMarkdownFiles(filePath, fileList);
        } else if (path.extname(file) === '.md') {
            fileList.push(filePath);
        }
    });

    return fileList;
}

// Function to remove <details> and </details> tags while keeping the content
function removeDetailsTags(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');

    // Regex to match <details> and </details> tags
    const detailsTagRegex = /<\/?details>/g;
    content = content.replace(detailsTagRegex, '');

    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`Processed file: ${filePath}`);
}

// Directory to search for markdown files
const directory = './docs/guides/';

const markdownFiles = getMarkdownFiles(directory);

markdownFiles.forEach(filePath => {
    removeDetailsTags(filePath);
});

console.log('All markdown files have been processed.');
