/* global chrome */
import React, {CSSProperties, Fragment, useEffect, useState} from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';
import axios from 'axios';
import RepoChat from './RepoChat';
import {ThreeDots, MutatingDots} from 'react-loader-spinner'
import {useNavigate} from 'react-router-dom';


const Home = () => {
  const navigate = useNavigate();
  const [containerSize, setContainerSize] = useState('w-520 h-400')

  const [readMeIsLoading, setReadMeIsLoading] = useState<boolean>(false)
  const [codeReviewIsLoading, setCodeReviewIsLoading] = useState<boolean>(false)
  const [temperature, setTemperature] = useState<number>(50)

  useEffect(() => {
    setContainerSize('w-250 h-200');
  }, []);

  const generateReadMe = () => {
    chrome.runtime.sendMessage({action: "generateReadMe"}, (response: any) => {
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
          console.log(username, repository)
          setReadMeIsLoading(true);
          axios.post(`http://127.0.0.1:5000/readme`,
              {
                username: username,
                repository: repository
              })
              .then(res => {
                setReadMeIsLoading(false)
                const fileContents = res.data.message
                console.log(fileContents)
                const element = document.createElement("a");
                const file = new Blob([fileContents], {type: 'text/markdown'});
                element.href = URL.createObjectURL(file);
                element.download = "README.md";
                document.body.appendChild(element);
                element.click();
                document.body.removeChild(element);
              })
        } else {
          alert('You must be a public github repository to generate ReadME\'s or ask about a repository.')
        }

      } else {
        console.error('No response received, or response was undefined.');
      }
    });
  };

  const generateCodeReview = () => {
    navigate('/codeReview')
  };

  return (
      <div className={`flex flex-col justify-center ${containerSize} items-center bg-gray-50 rounded-3xl transition-all duration-500 ease-in-out`}>
        <div className={'ml-4 w-full justify-center items-center max-w-xs bg-transparent pl-6 pt-4'}>
          <AnimatedTextGradient className={"justify-center font-bold text-3xl"} text={"Git. Read. Go."}></AnimatedTextGradient>
        </div>
        <div className="flex flex-col h-full justify-center items-center w-full">
          {
            !readMeIsLoading ?
                <button
                    className="w-150 h-8 bg-purple-700 hover:bg-purple-500 text-white font-bold mb-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105 disabled:opacity-50"
                    disabled={codeReviewIsLoading}
                    onClick={generateReadMe}
                >
                  Generate ReadME
                </button> :
                <ThreeDots
                    visible={true}
                    width="60"
                    color="#9436ff"
                    radius="9"
                    ariaLabel="three-dots-loading"
                    wrapperStyle={{}}
                    wrapperClass=""
                />
          }
          {
            !codeReviewIsLoading ?
                <button
                    className="bg-purple-700 w-150 h-8 hover:bg-purple-800 text-white font-bold mb-4 rounded transition duration-200 ease-in-out transform hover:-translate-y-1 hover:scale-105 disabled:opacity-50"
                    disabled={readMeIsLoading}
                    onClick={generateCodeReview}
                >
                  Code Review
                </button> :
                <ThreeDots
                    visible={true}
                    width="60"
                    color="#9436ff"
                    radius="9"
                    ariaLabel="three-dots-loading"
                    wrapperStyle={{}}
                    wrapperClass=""
                />
          }
        </div>
      </div>
  );
};

export default Home;
