/* global chrome */
import React, {CSSProperties, Fragment, useEffect, useState} from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';
import axios from 'axios';
import RepoChat from './RepoChat';
import {ThreeDots, MutatingDots} from 'react-loader-spinner'
import {useNavigate} from 'react-router-dom';


const CodeReview = () => {
  const navigate = useNavigate();
  const [containerSize, setContainerSize] = useState('w-250 h-200')

  useEffect(() => {
    setContainerSize('w-400 h-300');
  }, []);

  return (
      <div
          className={`flex flex-col items-center ${containerSize} bg-gray-50 rounded-3xl transition-all duration-500 ease-in-out`}>
        <div className={'flex flex-row w-full h-20'}>
          <div className={'flex flex-row items-center w-full max-w-xs bg-transparent pl-6 pt-4'}>
            <h1 className={'font-bold text-4xl mr-3'}>Score:</h1>
            <AnimatedTextGradient className={"font-black text-5xl"} text={'8/10'}></AnimatedTextGradient>
          </div>
        </div>
        <div className='flex flex-1 w-full'>
          <div className='flex flex-1 bg-red-300'>  {/* Flex-1 for each to fill available space horizontally */}
            {/* Additional content here */}
          </div>
          <div className='flex flex-1 bg-blue-300'>  {/* Flex-1 for each to fill available space horizontally */}
            {/* Additional content here */}
          </div>
        </div>
      </div>
  );
};

export default CodeReview;
