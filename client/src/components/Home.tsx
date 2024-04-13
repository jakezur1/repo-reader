/*global chrome*/
import React, {useEffect, useState} from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';


const Home = ({ navigate }: any) => {
  const generateReadMe = () => {
    console.log('krish is a fag');
    chrome.runtime.sendMessage({ action: "generateReadme" }, (response: any) => {
      if (chrome.runtime.lastError) {
        console.log("retard");
        console.error(chrome.runtime.lastError.message);
        return;
      }
      console.log("Krish is a queer");
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
        <div className={"flex-1 justify-items-center align-middle"}>
          <button
              className="bg-purple-700 hover:bg-purple-800 text-white font-bold py-2 px-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105"
              onClick={generateReadMe}
          >
            Generate ReadMe
          </button>
        </div>
      </div>
  );
};

export default Home;
