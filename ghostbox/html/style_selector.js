document.addEventListener('DOMContentLoaded', (event) => {
    const stylesheetSelector = document.getElementById('stylesheetSelector');
    const defaultStylesheet = 'styles/sakura-dark-solarized.css';

    // Function to load stylesheet names from active_styles file
    function loadStylesheetNames() {
        return fetch('active_styles')
            .then(response => response.text())
            .then(data => data.split('\n').filter(Boolean)); // Split by newline and filter out empty lines
    }

    // Load the saved stylesheet from local storage
    const savedStylesheet = localStorage.getItem('selectedStylesheet');
    const stylesheetToApply = savedStylesheet || defaultStylesheet;

    // Apply the selected stylesheet
    applyStylesheet(stylesheetToApply);

    // Load stylesheet names and populate the dropdown
    loadStylesheetNames().then(stylesheetNames => {
        stylesheetNames.forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name.replace('.css', ''); // Remove .css extension for display
            stylesheetSelector.appendChild(option);
        });

        // Set the selected stylesheet if saved in local storage
        if (savedStylesheet) {
            stylesheetSelector.value = savedStylesheet.split('/').pop();
        }

        // Add event listener for stylesheet selection
        stylesheetSelector.addEventListener('change', (event) => {
            const selectedStylesheet = 'styles/' + event.target.value;
            applyStylesheet(selectedStylesheet);
            // Save the selected stylesheet to local storage
            localStorage.setItem('selectedStylesheet', selectedStylesheet);
        });
    });

    function applyStylesheet(stylesheet) {
        // Remove any existing stylesheet links
        const existingLink = document.querySelector('link[rel="stylesheet"][href^="styles/"]');
        if (existingLink) {
            document.head.removeChild(existingLink);
        }

        // Create and append the new stylesheet link
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = stylesheet;
        document.head.appendChild(link);
    }
});

