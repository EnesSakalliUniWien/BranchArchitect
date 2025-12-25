CSS_LOG = """
/* Base styles */
.split-analysis {
    margin-top: 1em;
    padding: 1em;
    background: #2d2d2d;
    border-radius: 4px;
}

.error-detail {
    margin: 1em 0;
    padding: 1em;
    background: #331f1f;
    border-radius: 4px;
}

.error-detail p {
    margin: 0.5em 0;
}

.error-detail strong {
    color: #ff8080;
}

/* MathJax matrix container - primary display */
.mathjax-matrix {
    margin: 2em 0;
    padding: 20px;
    background: #1e293b;
    border-radius: 8px;
    border: 1px solid #3a4a6d;
    text-align: center;
    overflow-x: auto;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.mathjax-matrix .MJX-TEX {
    font-size: 120% !important;
}

/* Matrix container and views */
.matrix-container {
    margin: 2em 0;
    position: relative;
}

/* Matrix view toggle controls */
.matrix-toggle {
    margin: 1.5em 0 1em;
    text-align: right;
}

.toggle-buttons {
    display: inline-flex;
    background: #2c3445;
    border-radius: 6px;
    padding: 4px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

.toggle-button {
    background: transparent;
    border: none;
    color: #8a9bbc;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'Menlo', 'DejaVu Sans Mono', monospace;
    font-size: 0.9em;
    transition: all 0.2s ease;
}

.toggle-button:hover {
    color: #dbe4fd;
    background: #3a4559;
}

.toggle-button.active {
    background: #394963;
    color: #ffffff;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

/* ASCII Matrix (shown when toggled) */
.ascii-matrix {
    margin: 1em 0;
    padding: 25px 30px;
    background: #282c34;
    border-radius: 8px;
    border-left: 4px solid #61afef;
}

.ascii-matrix pre {
    font-family: 'DejaVu Sans Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 1.1em;
    line-height: 1.5;
    margin: 0;
    color: #e0e0e0;
    white-space: pre;
}

/* Matrix views */
.mathjax-view {
    padding: 20px 30px;
    background: #1c2333;
    background: linear-gradient(145deg, #1a1e2d 0%, #252b3b 100%);
    border-radius: 8px;
    border: 1px solid #3a4a6d;
    text-align: center;
    overflow-x: auto;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.mathjax-view .MJX-TEX {
    font-size: 120% !important;
}

.ascii-view pre {
    background: #282c34;
    padding: 15px;
    border-radius: 5px;
    font-family: 'DejaVu Sans Mono', monospace;
    overflow-x: auto;
    white-space: pre;
}

/* Bracket and element coloring */
.bracket {
    color: #e5c07b;
    font-weight: bold;
    font-size: 1.4em;
}

.set {
    color: #98c379;
}

.element {
    color: #61afef;
}

.matrix-divider, .divider {
    color: #636b7c;
}
"""


COMPARE_TREE_SPLIT_CSS = """
    .split-comparison {
        margin: 20px 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .controls {
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .sort-options, .column-toggle {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .sort-btn, .col-toggle-btn {
        background: #333;
        color: #ccc;
        border: 1px solid #555;
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .sort-btn:hover, .col-toggle-btn:hover {
        background: #444;
        color: #fff;
    }
    .sort-btn.active {
        background: #007acc;
        color: white;
    }
    .col-toggle-btn.active {
        background: #2c5e2e;
        color: white;
    }
    .table-wrapper {
        max-height: 600px;
        overflow-y: auto;
        border-radius: 5px;
        border: 1px solid #444;
    }
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #222;
        color: #ccc;
    }
    .comparison-table th, .comparison-table td {
        padding: 8px 12px;
        text-align: left;
        border: 1px solid #444;
    }
    .comparison-table th {
        position: sticky;
        top: 0;
        background-color: #333;
        z-index: 10;
        color: white;
    }
    .comparison-table tr.common {
        background-color: rgba(0, 128, 0, 0.1);
    }
    .comparison-table tr.different {
        background-color: rgba(128, 0, 0, 0.1);
    }
    .comparison-table tr:hover {
        background-color: #2a2a2a;
    }
    .comparison-table td.match {
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
    }
    .comparison-table td.mismatch {
        color: #F44336;
        font-weight: bold;
        text-align: center;
    }
    .summary-box {
        margin-top: 15px;
        padding: 10px 15px;
        background-color: #2a2a2a;
        border-left: 4px solid #007acc;
        border-radius: 3px;
    }
    /* Hide columns when toggled off */
    .comparison-table .col-hidden {
        display: none;
    }
"""


TABLE_SPLIT_JS = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    const splitComparisons = document.querySelectorAll('.split-comparison');
    
    splitComparisons.forEach(comparison => {
        // Sort functionality
        const sortButtons = comparison.querySelectorAll('.sort-btn');
        const tableBody = comparison.querySelector('tbody');
        const rows = Array.from(tableBody.querySelectorAll('tr'));
        
        // Save original order
        const originalOrder = [...rows];
        
        sortButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                // Update active button
                sortButtons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                const sortType = this.getAttribute('data-sort');
                
                // Sort rows
                let newRows;
                if (sortType === 'indices') {
                    newRows = [...rows].sort((a, b) => {
                        const aIndices = a.cells[0].textContent;
                        const bIndices = b.cells[0].textContent;
                        return aIndices.localeCompare(bIndices);
                    });
                } else if (sortType === 'taxa') {
                    newRows = [...rows].sort((a, b) => {
                        const aTaxa = a.cells[1].textContent;
                        const bTaxa = b.cells[1].textContent;
                        return aTaxa.localeCompare(bTaxa);
                    });
                } else if (sortType === 'common') {
                    newRows = [...rows].sort((a, b) => {
                        const aCommon = a.classList.contains('common') ? 0 : 1;
                        const bCommon = b.classList.contains('common') ? 0 : 1;
                        return aCommon - bCommon;
                    });
                } else if (sortType === 'diff') {
                    newRows = [...rows].sort((a, b) => {
                        const aCommon = a.classList.contains('common') ? 0 : 1;
                        const bCommon = b.classList.contains('common') ? 0 : 1;
                        return bCommon - aCommon;
                    });
                } else {
                    newRows = originalOrder;
                }
                
                // Update table
                tableBody.innerHTML = '';
                newRows.forEach(row => tableBody.appendChild(row));
            });
        });
        
        // Column toggle functionality
        const colToggleButtons = comparison.querySelectorAll('.col-toggle-btn');
        const table = comparison.querySelector('table');
        
        colToggleButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                // Toggle active state
                this.classList.toggle('active');
                
                const colIndex = this.getAttribute('data-col');
                const isActive = this.classList.contains('active');
                
                // Toggle visibility of column header
                const headers = table.querySelectorAll('th');
                if (headers[colIndex]) {
                    headers[colIndex].classList.toggle('col-hidden', !isActive);
                }
                
                // Toggle visibility of column cells
                const cells = table.querySelectorAll(`td.col-${colIndex}`);
                cells.forEach(cell => {
                    cell.classList.toggle('col-hidden', !isActive);
                });
            });
        });
    });
});
</script>
"""


MATH_JAX_HEADER = """
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true,
    packages: ['base', 'ams', 'noerrors', 'noundefined']
  },
  svg: {
    fontCache: 'global',
    scale: 1.1
  },
  startup: {
    typeset: true
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""


UNIQUE_ATOMS_COVETS_CODE_CSS = """
.code-container {
    margin: 15px 0;
    border: 1px solid #444;
    border-radius: 5px;
    overflow: hidden;
    background: #1c2333;
}
.code-header {
    background: #2c3445;
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #3a4a6d;
}
.copy-button {
    background: #394963;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    cursor: pointer;
    transition: background 0.2s;
}
.copy-button:hover {
    background: #4a5d7e;
}
.code-block {
    margin: 0;
    padding: 15px;
    background: #1a1e2d;
    overflow-x: auto;
    color: #e0e0e0;
    font-family: 'Fira Code', 'DejaVu Sans Mono', 'Courier New', monospace;
}
"""

# Newick visualization CSS styled to match the dark theme
NEWICK_VISUALIZATION_CSS = """
.newick-container {
    margin-bottom: 20px;
    font-family: 'DejaVu Sans Mono', 'Fira Code', 'Courier New', monospace;
}
.newick-box {
    border: 1px solid #444;
    border-radius: 5px;
    margin: 10px 0;
    background-color: #1c2333;
    position: relative;
}
.newick-header {
    background-color: #2c3445;
    padding: 8px 12px;
    border-bottom: 1px solid #3a4a6d;
    font-weight: bold;
    border-radius: 5px 5px 0 0;
    display: flex;
    justify-content: space-between;
    color: #e0e0e0;
}
.newick-content {
    padding: 12px;
    word-break: break-all;
    white-space: pre-wrap;
    max-height: 150px;
    overflow-y: auto;
    color: #e0e0e0;
    background-color: #1a1e2d;
}
.copy-btn {
    background-color: #394963;
    color: white;
    border: none;
    border-radius: 3px;
    padding: 3px 8px;
    font-size: 12px;
    cursor: pointer;
    transition: background-color 0.2s;
}
.copy-btn:hover {
    background-color: #4a5d7e;
}
.copy-success {
    background-color: #28a745 !important;
}
"""

# Newick JS for copy functionality
NEWICK_COPY_JS = """
<script>
function copyNewick(button, elementId) {
    const newickText = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(newickText).then(() => {
        button.textContent = "Copied!";
        button.classList.add("copy-success");
        setTimeout(() => {
            button.textContent = "Copy";
            button.classList.remove("copy-success");
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        button.textContent = "Failed";
        setTimeout(() => {
            button.textContent = "Copy";
        }, 2000);
    });
}
</script>
"""

# HTML template for Newick tree visualization
NEWICK_TEMPLATE_ONE_TREE = """
<div class="newick-container">
    <div class="newick-box">
        <div class="newick-header">
            <span>Tree 1</span>
            <button class="copy-btn" onclick="copyNewick(this, 'newick1')">Copy</button>
        </div>
        <div class="newick-content" id="newick1">{0}</div>
    </div>
</div>
"""

# Page-level base CSS for debug HTML documents
DEBUG_PAGE_CSS = """
/* Reset and base styles */
* { margin: 0; padding: 0; box-sizing: border-box; }

/* Core theme */
body {
    background: #1d1d1d;
    color: #c8c8c8;
    font-family: 'Menlo', 'DejaVu Sans Mono', monospace;
    line-height: 1.6;
    padding: 20px;
    -webkit-font-smoothing: antialiased;
}

/* Layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    background: #222;
    border: 1px solid #333;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #fff;
    margin: 1em 0 0.5em 0;
    border-bottom: 1px solid #333;
    padding-bottom: 0.3em;
}

/* Tables */
.table-container {
    margin: 1em 0;
    overflow-x: auto;
    background: #252525;
    border-radius: 4px;
    border: 1px solid #333;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
    background: transparent;
}

th {
    background: #333;
    color: #fff;
    font-weight: bold;
    text-align: left;
    padding: 12px;
    border: 1px solid #444;
}

td {
    padding: 10px 12px;
    border: 1px solid #444;
    color: #c8c8c8;
}

tr:hover {
    background: #2a2a2a;
}

/* SVG/IMG containers */
.svg-container {
    background: #252525;
    padding: 20px;
    margin: 1em 0;
    border-radius: 4px;
    border: 1px solid #333;
    overflow-x: auto;
}

.svg-container svg,
.svg-container img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}

/* Analysis cards */
.bidirectional-analysis {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 1em 0;
}

.analysis-card {
    background: #252525;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 4px;
}

.analysis-detail {
    margin: 8px 0;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #333;
    padding-bottom: 4px;
}

.detail-label {
    color: #888;
    margin-right: 10px;
}

/* Messages */
.warning { color: #ffd700; }
.error { color: #ff6b6b; }
.success { color: #6bff6b; }

/* Code blocks */
pre {
    background: #252525;
    padding: 15px;
    border-radius: 4px;
    overflow-x: auto;
    border: 1px solid #333;
    margin: 1em 0;
}

/* Links */
a {
    color: #4a9eff;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}
"""

NEWICK_TEMPLATE_TWO_TREES = """
<div class="newick-container">
    <div class="newick-box">
        <div class="newick-header">
            <span>Tree 1</span>
            <button class="copy-btn" onclick="copyNewick(this, 'newick1')">Copy</button>
        </div>
        <div class="newick-content" id="newick1">{0}</div>
    </div>
    <div class="newick-box">
        <div class="newick-header">
            <span>Tree 2</span>
            <button class="copy-btn" onclick="copyNewick(this, 'newick2')">Copy</button>
        </div>
        <div class="newick-content" id="newick2">{1}</div>
    </div>
</div>
"""

# Add bracket highlighting JS for Newick strings
NEWICK_HIGHLIGHT_JS = """
<script>
function highlightBrackets(containerId) {
    const container = document.getElementById(containerId);
    const text = container.textContent;
    
    // Create a new structure with highlighted brackets
    let newHtml = '';
    const stack = [];
    const bracketPairs = [];
    const colors = [
        "#e84393", "#00cec9", "#0984e3", "#6c5ce7", 
        "#fdcb6e", "#e17055", "#00b894", "#74b9ff",
        "#a29bfe", "#55efc4", "#fab1a0", "#81ecec"
    ];
    
    // First pass: identify bracket pairs
    for (let i = 0; i < text.length; i++) {
        if (text[i] === '(') {
            stack.push(i);
        } else if (text[i] === ')') {
            if (stack.length > 0) {
                const openPos = stack.pop();
                bracketPairs.push({ open: openPos, close: i, level: stack.length % colors.length });
            }
        }
    }
    
    // Second pass: build HTML with colored brackets
    let lastPos = 0;
    bracketPairs.sort((a, b) => a.open - b.open);
    
    for (const pair of bracketPairs) {
        // Add text before opening bracket
        if (pair.open > lastPos) {
            newHtml += text.substring(lastPos, pair.open);
        }
        
        // Add opening bracket with color
        newHtml += `<span class="bracket" style="color:${colors[pair.level]}">${text[pair.open]}</span>`;
        
        // Add content between brackets
        const innerContent = text.substring(pair.open + 1, pair.close);
        newHtml += innerContent;
        
        // Add closing bracket with color
        newHtml += `<span class="bracket" style="color:${colors[pair.level]}">${text[pair.close]}</span>`;
        
        lastPos = pair.close + 1;
    }
    
    // Add any remaining text
    if (lastPos < text.length) {
        newHtml += text.substring(lastPos);
    }
    
    container.innerHTML = newHtml;
}

function copyAllNewick(button) {
    const newickContent = document.getElementById('newick-combined').textContent;
    navigator.clipboard.writeText(newickContent).then(() => {
        button.textContent = "Copied!";
        button.classList.add("copy-success");
        setTimeout(() => {
            button.textContent = "Copy All";
            button.classList.remove("copy-success");
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        button.textContent = "Failed";
        setTimeout(() => {
            button.textContent = "Copy All";
        }, 2000);
    });
}
</script>
"""

# Update the Newick visualization CSS for combined box
NEWICK_COMBINED_CSS = """
.newick-container {
    margin-bottom: 20px;
    font-family: 'DejaVu Sans Mono', 'Fira Code', 'Courier New', monospace;
}
.newick-box {
    border: 1px solid #444;
    border-radius: 5px;
    margin: 10px 0;
    background-color: #1c2333;
    position: relative;
}
.newick-header {
    background-color: #2c3445;
    padding: 8px 12px;
    border-bottom: 1px solid #3a4a6d;
    font-weight: bold;
    border-radius: 5px 5px 0 0;
    display: flex;
    justify-content: space-between;
    color: #e0e0e0;
}
.newick-content {
    padding: 12px;
    word-break: break-all;
    white-space: pre-wrap;
    max-height: 150px;
    overflow-y: auto;
    color: #e0e0e0;
    background-color: #1a1e2d;
}
.combined-box {
    border: 1px solid #444;
    border-radius: 5px;
    margin: 15px 0;
    background-color: #1c2333;
}
.combined-header {
    background-color: #2c3445;
    padding: 10px 15px;
    border-bottom: 1px solid #3a4a6d;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    color: #e0e0e0;
    align-items: center;
}
.combined-content {
    padding: 15px;
    background-color: #1a1e2d;
    color: #e0e0e0;
    overflow-x: auto;
    white-space: pre-wrap;
}
.tree-label {
    color: #61afef;
    font-weight: bold;
    margin-right: 5px;
}
.copy-btn, .copy-all-btn {
    background-color: #394963;
    color: white;
    border: none;
    border-radius: 3px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background-color 0.2s;
}
.copy-btn:hover, .copy-all-btn:hover {
    background-color: #4a5d7e;
}
.copy-success {
    background-color: #28a745 !important;
}
.bracket {
    font-weight: bold;
}
"""

# HTML template for the combined copyable Newick box
NEWICK_TEMPLATE_COMBINED = """
<div class="combined-box">
    <div class="combined-header">
        <span>Newick Representation</span>
        <button class="copy-all-btn" onclick="copyAllNewick(this)">Copy All</button>
    </div>
    <div class="combined-content" id="newick-combined"><span class="tree-label">Tree1:</span> {0}

<hr style="border: none; border-top: 1px dashed #4a5d7e; margin: 12px 0;">

<span class="tree-label">Tree2:</span> {1}</div>
</div>
"""

NEWICK_TEMPLATE_SINGLE_COMBINED = """
<div class="combined-box">
    <div class="combined-header">
        <span>Newick Representation</span>
        <button class="copy-all-btn" onclick="copyAllNewick(this)">Copy All</button>
    </div>
    <div class="combined-content" id="newick-combined">Tree: {0}</div>
</div>
"""
