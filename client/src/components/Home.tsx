import React from 'react';

const Home = ({ navigate }: any) => {
  const generateReadMe = () => {
    // Sending a message to the background script
    chrome.runtime.sendMessage({ action: "generateReadme" }, (response) => {
      if (chrome.runtime.lastError) {
        // Handle error, for example:
        console.error(chrome.runtime.lastError.message);
        return;
      }
      
      // Ensure the response exists before trying to access its properties
      if (response) {
        console.log("Response from background:", response.status);
      } else {
        console.error('No response received, or response was undefined.');
      }
    });
  };

  return (
    <div className="w-300 h-300 rounded-2rem">
      <div className="bg-transparent">
        {/*<AnimatedLinearGradient customColors={presetColors.instagram} speed={4000}/>*/}
        <h1 className="text-4xl m-3 font-bold text-purple-700">Welcome to RepoReader</h1>
      </div>
      <button 
        className="bg-purple-700 hover:bg-purple-800 text-white font-bold py-2 px-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105"
        onClick={generateReadMe}
      >
        Generate ReadMe
      </button>
    </div>
  );
};

export default Home;
