/* global chrome */
import React, {CSSProperties, Fragment, useEffect, useState} from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';
import axios from 'axios';
import {MutatingDots} from 'react-loader-spinner'


const Home = ({navigate}: any) => {

  const [readMeIsLoading, setReadMeIsLoading] = useState<boolean>(false)

  const generateReadMe = () => {
    chrome.runtime.sendMessage({action: "generateReadme"}, (response: any) => {
      if (chrome.runtime.lastError) {
        console.error(chrome.runtime.lastError.message);
        return;
      }
      if (response) {
        const url = response.message
        if (url.includes("github")) {
          const pathSegments = new URL(url).pathname.split('/').filter(Boolean);
          const username = pathSegments[0];
          const repository = pathSegments[1];
          setReadMeIsLoading(true);
          axios.post(`http://127.0.0.1:5000/readme`,
              {
                username: username,
                repository: repository
              }, {
                headers: {
                  'Content-Type': 'application/json',
                  'Access-Control-Allow-Origin': 'chrome-extension://lckojlmkgfdgahdmpjddkbonggndjobi'
                }
              })
              .then(res => {
                console.log('hello')
                setReadMeIsLoading(false)
                // chrome.runtime.sendMessage({action: "downloadReadMe", data: ''}, (response: any) => {
                //   // send file
                // })
              }).catch(() => {
            console.log('hi')
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
          {
            !readMeIsLoading ?
                <button
                    className="bg-purple-700 hover:bg-purple-800 text-white font-bold mb-4 py-2 px-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105"
                    onClick={generateReadMe}
                >
                  Generate ReadME
                </button> :
                <MutatingDots
                    visible={readMeIsLoading}
                    height="100"
                    width="100"
                    color="#4c00b5"
                    secondaryColor="#a463ff"
                    radius="10"
                    ariaLabel="mutating-dots-loading"
                    wrapperStyle={{}}
                    wrapperClass=""
                />
          }
          <button
              className="bg-purple-700 hover:bg-purple-800 text-white font-bold mb-4 py-2 px-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105 disabled:opacity-50"
              disabled={readMeIsLoading}
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
