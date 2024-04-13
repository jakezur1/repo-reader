/*global chrome*/
// This is a simple message handler that responds to messages with a predefined action.

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Check the action property to determine what to do.
  if (message.action === "generateReadme") {
    // Your logic to handle the 'generateReadme' action goes here.
    const url = logCurrentTabUrl();

    // Send a response back to the sender.
    console.log(url)
    sendResponse({ message: url });
  } else {
    // Handle any other actions, or send a response indicating the action was not recognized.
    console.log(`Unrecognized action: ${message.action}`);
    sendResponse({ status: "Unrecognized Action" });
  }

  // Return true to indicate you wish to send a response asynchronously
  return true;
});

// Function to log the URL of the current active tab
function logCurrentTabUrl() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    var currentTab = tabs[0];
    if (currentTab && currentTab.url) {
      return currentTab.url;
    } else {
      console.log('No active tab or URL found.');
    }
  });
  return "";
}

// Depending on what your extension does, you might not need to listen to browser action clicks
// If you do, this is correct.
chrome.action.onClicked.addListener((tab) => {
  logCurrentTabUrl();
});
