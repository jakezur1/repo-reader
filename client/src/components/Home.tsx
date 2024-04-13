/*global chrome*/
import React from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';


const Home = ({ navigate }: any) => {
  const generateReadMe = () => {
    // Sending a message to the background script
    chrome.runtime.sendMessage({ action: "generateReadme" }, (response: any) => {
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
      <div className={"w-300 h-300 bg-gray-50 rounded border-4 border-purple-700 m-2"}>
        <div className={'bg-transparent p-3'}>
          <AnimatedTextGradient text={"Welcome to RepoReader!"}></AnimatedTextGradient>
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
