/**
 * Matrix visualization utilities
 */

document.addEventListener('DOMContentLoaded', function() {
    // Wait for MathJax to be defined
    function checkMathJax() {
        if (typeof MathJax !== 'undefined') {
            enhanceMatrixDisplay();
        } else {
            setTimeout(checkMathJax, 100);
        }
    }
    
    checkMathJax();
});

/**
 * Enhances the display of matrices by adding interactive elements
 * and consistent styling
 */
function enhanceMatrixDisplay() {
    // Add toggle buttons to switch between different matrix views
    const matrixContainers = document.querySelectorAll('.mathjax-matrix, .ascii-matrix, .matrix-table');
    
    // Group matrix containers by their parent section
    const matrixSets = {};
    matrixContainers.forEach(container => {
        // Find the nearest heading as an identifier
        let heading = container.previousElementSibling;
        while (heading && heading.tagName.toLowerCase() !== 'h4') {
            heading = heading.previousElementSibling;
        }
        
        const headingText = heading ? heading.textContent : 'Matrix';
        if (!matrixSets[headingText]) {
            matrixSets[headingText] = [];
        }
        matrixSets[headingText].push(container);
    });
    
    // Add toggle controls for each set of matrices
    Object.keys(matrixSets).forEach(key => {
        const containers = matrixSets[key];
        if (containers.length <= 1) return;
        
        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'matrix-toggle';
        toggleContainer.innerHTML = `
            <div class="toggle-buttons">
                <button class="toggle-button active" data-view="mathjax">Mathematical</button>
                <button class="toggle-button" data-view="ascii">ASCII Art</button>
                <button class="toggle-button" data-view="table">Detailed Table</button>
            </div>
        `;
        
        // Insert toggle before the first container
        containers[0].parentNode.insertBefore(toggleContainer, containers[0]);
        
        // Initialize view (show only MathJax view by default)
        containers.forEach(container => {
            if (container.classList.contains('mathjax-matrix')) {
                container.style.display = 'block';
            } else {
                container.style.display = 'none';
            }
        });
        
        // Add event listeners to toggle buttons
        const buttons = toggleContainer.querySelectorAll('.toggle-button');
        buttons.forEach(button => {
            button.addEventListener('click', function() {
                // Update active button
                buttons.forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                
                // Show/hide appropriate container
                const view = this.getAttribute('data-view');
                containers.forEach(container => {
                    if (container.classList.contains(`${view}-matrix`) ||
                        (view === 'table' && container.classList.contains('matrix-table'))) {
                        container.style.display = 'block';
                    } else {
                        container.style.display = 'none';
                    }
                });
                
                // Trigger MathJax re-render if switching to MathJax view
                if (view === 'mathjax' && typeof MathJax !== 'undefined') {
                    MathJax.typeset();
                }
            });
        });
    });
    
    // Add hover effects for matrix cells in tables
    const matrixCells = document.querySelectorAll('.matrix-table td');
    matrixCells.forEach(cell => {
        // Skip first column (row labels)
        if (cell.cellIndex === 0) return;
        
        cell.addEventListener('mouseenter', function() {
            // Highlight this cell
            this.classList.add('highlight');
            
            // Highlight corresponding row and column
            const rowIndex = this.parentNode.rowIndex;
            const colIndex = this.cellIndex;
            
            const table = this.closest('table');
            const rowCells = table.rows[rowIndex].cells;
            const colCells = Array.from(table.rows).map(row => row.cells[colIndex]);
            
            Array.from(rowCells).forEach(c => c.classList.add('row-highlight'));
            colCells.forEach(c => c.classList.add('col-highlight'));
        });
        
        cell.addEventListener('mouseleave', function() {
            // Remove all highlights
            document.querySelectorAll('.highlight, .row-highlight, .col-highlight')
                .forEach(el => {
                    el.classList.remove('highlight');
                    el.classList.remove('row-highlight');
                    el.classList.remove('col-highlight');
                });
        });
    });
    
    // Run MathJax typesetting again to ensure all math is properly rendered
    MathJax.typeset();
}
