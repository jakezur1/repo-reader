(()=>{function e(){chrome.tabs.query({active:!0,currentWindow:!0},(e=>{var o=e[0];o&&o.url?console.log(o.url):console.log("No active tab or URL found.")}))}chrome.runtime.onMessage.addListener(((o,n,t)=>(console.log("ni"),"generateReadme"===o.action?(console.log("hi"),e(),console.log("Received generateReadme action"),t({status:"ReadMe Generation Started"})):(console.log(`Unrecognized action: ${o.action}`),t({status:"Unrecognized Action"})),!0))),chrome.action.onClicked.addListener((o=>{e()}))})();