/**
 * TOC Code Style Script
 *
 * Applies monospace/code styling to CLI command entries in the right-side
 * table of contents (TOC). MkDocs Material strips inline code formatting
 * from headings when generating TOC entries, so this script re-applies
 * the styling after page load.
 *
 * Targets entries like:
 *   - sio show
 *   - sio convert
 *   - sio split
 *   etc.
 */

/**
 * Apply code styling to CLI command TOC entries.
 */
function styleTocCommands() {
  // Target TOC links in the right-side navigation (secondary nav)
  // Material theme uses .md-nav--secondary for the right TOC
  const tocLinks = document.querySelectorAll(
    '.md-nav--secondary .md-nav__link, .md-sidebar--secondary .md-nav__link'
  );

  tocLinks.forEach(function (link) {
    const href = link.getAttribute('href');
    // Match links like #sio-show, #sio-convert, etc.
    if (href && /^#sio-/.test(href)) {
      // Apply monospace styling
      link.style.fontFamily =
        'var(--md-code-font, "Roboto Mono"), SFMono-Regular, Consolas, Menlo, monospace';
      link.style.fontSize = '0.85em';
      // Optional: add a subtle background like inline code
      // link.style.backgroundColor = 'var(--md-code-bg-color)';
      // link.style.padding = '0.1em 0.2em';
      // link.style.borderRadius = '0.2em';
    }
  });
}

// Material theme with instant loading requires using document$ observable
// This ensures the script runs on initial load AND when navigating between pages
if (typeof document$ !== 'undefined') {
  // Material theme instant loading mode
  document$.subscribe(function () {
    styleTocCommands();
  });
} else {
  // Fallback for non-instant loading or initial page load
  document.addEventListener('DOMContentLoaded', styleTocCommands);

  // Also run immediately in case DOM is already loaded
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    styleTocCommands();
  }
}
