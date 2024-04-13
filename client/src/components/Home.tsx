/* global chrome */
import React, {Fragment, useEffect, useState} from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';
import axios from 'axios';
import GridLoader from "react-spinners/GridLoader";


const Home = ({navigate}: any) => {

  const [isLoading, setIsLoading] = useState<boolean>(false)

  const generateReadMe = () => {
    chrome.runtime.sendMessage({action: "generateReadme"}, (response: any) => {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError.message);
        return;
      }
      if (response) {
        const url = response.message
        console.log("Response from background:", url);
        console.log(url)
        if (url.includes("github")) {
          const pathSegments = new URL(url).pathname.split('/').filter(Boolean);
          const username = pathSegments[1];
          const repository = pathSegments[2];
          setIsLoading(true);
          axios.post(`http://127.0.0.1:5000/readme`, {username: username, repo: repository})
              .then(res => {
                setIsLoading(false)
                chrome.runtime.sendMessage({action: "downloadReadMe", data: ''}, (response: any) => {
                  // send file
                })
              })
        } else {
          alert('You must be a public github repository to generate ReadME\'s or ask about a repository.')
        }

      } else {
        console.error('No response received, or response was undefined.');
      }
    });
  };

  return (
      <div className={"flex flex-col items-center w-300 h-300 bg-gray-50 rounded border-4 border-purple-700 m-2"}>
        <div className={'w-full max-w-xs bg-transparent p-3'}>
          <AnimatedTextGradient text={"Welcome to RepoRead!"}></AnimatedTextGradient>
        </div>
        <div className="flex flex-col h-full justify-center items-center w-full">
          <button
              className="bg-purple-700 hover:bg-purple-800 text-white font-bold mb-4 py-2 px-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105"
              onClick={generateReadMe}
          >
            {
              !isLoading ? <h1>Generate ReadME</h1> :
                  <GridLoader
                      color={"%6f00ff"}
                      loading={isLoading}
                      size={150}
                      aria-label="Loading Spinner"
                      data-testid="loader"
                  />
            }
          </button>
          <button
              className="bg-purple-700 hover:bg-purple-800 text-white font-bold mb-4 py-2 px-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105"
              onClick={() => {
              }}
          >
            Repo Chat
          </button>
        </div>
      </div>
  );
};

export default Home;
