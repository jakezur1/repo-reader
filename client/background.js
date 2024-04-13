/*global chrome*/
// This is a simple message handler that responds to messages with a predefined action.

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Check the action property to determine what to do.
  // console.log("ni");
  if (message.action === "generateReadme") {
    // Your logic to handle the 'generateReadme' action goes here.
    getCurrentTabUrl().then(url => {
      sendResponse({ message: url });
    }).catch(error => {
      console.error("Error:", error);
    });
  } 
  else if(message.action == "downloadReadMe"){
    const data = message.data ;
    const blob = new Blob([data], {type: 'text/plain'});
    const url = URL.createObjectURL(blob);
    chrome.downloads.download({
      url: url,
      filename: "myReadMe.txt",
      conflictAction: 'uniquify'
    }, (downloadId) => {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError);
      } else {
        console.log(`Download initiated with ID: ${downloadId}`);
      }
    });
  }
  else {
    // Handle any other actions, or send a response indicating the action was not recognized.
    console.log(`Unrecognized action: ${message.action}`);
    sendResponse({ status: "Unrecognized Action" });
  }

  // Return true to indicate you wish to send a response asynchronously
  return true;
});

// Function to log the URL of the current active tab
function getCurrentTabUrl() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      var currentTab = tabs[0];
      if (currentTab && currentTab.url) {
        resolve((currentTab.url));
      } else {
        console.log('No active tab or URL found.');
      }
    })
  });
}

// Depending on what your extension does, you might not need to listen to browser action clicks
// If you do, this is correct.
chrome.action.onClicked.addListener((tab) => {
  getCurrentTabUrl();
});
