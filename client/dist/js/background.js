<<<<<<< Updated upstream
(()=>{function e(){return new Promise(((e,o)=>{chrome.tabs.query({active:!0,currentWindow:!0},(o=>{var n=o[0];n&&n.url?e(n.url):console.log("No active tab or URL found.")}))}))}chrome.runtime.onMessage.addListener(((o,n,r)=>("getURL"===o.action?e().then((e=>{r({message:e})})).catch((e=>{console.error("Error:",e)})):(console.log(`Unrecognized action: ${o.action}`),r({status:"Unrecognized Action"})),!0))),chrome.action.onClicked.addListener((o=>{e()}))})();
=======
(()=>{function e(){return new Promise(((e,o)=>{chrome.tabs.query({active:!0,currentWindow:!0},(o=>{var n=o[0];n&&n.url?e(n.url):console.log("No active tab or URL found.")}))}))}chrome.runtime.onMessage.addListener(((o,n,r)=>(console.log("ni"),"generateReadMe"===o.action?e().then((e=>{e.includes("github")?r({message:e}):(chrome.notifications.create("no_github_found",{type:"basic",iconUrl:"icons/icon128.png",title:"No Repository Found",message:"You must be on a public GitHub repository to generate READMEs or request a Code Review.",priority:2}),r({message:""}))})).catch((e=>{console.error("Error:",e)})):(console.log(`Unrecognized action: ${o.action}`),r({status:"Unrecognized Action"})),!0))),chrome.action.onClicked.addListener((o=>{e()}))})();
>>>>>>> Stashed changes
