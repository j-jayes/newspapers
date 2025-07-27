/**
 * This script finds all '.browser-window' elements in a reveal.js presentation
 * and sets the title bar text to the content of the image's figcaption.
 * v3: Uses a MutationObserver for a more reliable execution.
 */

// This function performs the actual update on a given window.
function updateSingleBrowserWindow(win) {
  // Check if this window has already been processed.
  if (win.dataset.captionUpdated) {
    return;
  }

  // Find the caption within the container.
  const figcaption = win.querySelector('figcaption');

  if (figcaption && figcaption.textContent) {
    // Get the caption text and clean it up.
    const captionText = figcaption.textContent.trim();

    // Set the CSS variable on the element for the title bar.
    win.style.setProperty('--browser-title', `'${captionText}'`);

    // Hide the original caption.
    figcaption.style.display = 'none';

    // Mark this window as processed.
    win.dataset.captionUpdated = 'true';
  }
}

// This function initializes the observer.
function initializeBrowserCaptionObserver() {
  const slidesContainer = document.querySelector('.reveal .slides');

  if (!slidesContainer) {
    // If the slides container isn't ready, try again in a moment.
    window.setTimeout(initializeBrowserCaptionObserver, 100);
    return;
  }

  // 1. Run the update on any browser windows that might already exist.
  slidesContainer.querySelectorAll('.browser-window').forEach(updateSingleBrowserWindow);

  // 2. Create an observer to watch for future changes.
  const observer = new MutationObserver((mutationsList) => {
    for (const mutation of mutationsList) {
      if (mutation.type === 'childList') {
        // Check if any new nodes are or contain a .browser-window
        mutation.addedNodes.forEach(node => {
          if (node.nodeType === 1) { // Ensure it's an element
            if (node.classList.contains('browser-window')) {
              updateSingleBrowserWindow(node);
            } else {
              // Also check descendants of the new node
              node.querySelectorAll('.browser-window').forEach(updateSingleBrowserWindow);
            }
          }
        });
      }
    }
  });

  // 3. Start observing the slides container for added/removed children.
  observer.observe(slidesContainer, { childList: true, subtree: true });
}

// Start the process.
initializeBrowserCaptionObserver();
