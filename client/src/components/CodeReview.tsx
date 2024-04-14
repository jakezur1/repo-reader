/* global chrome */
import React, {CSSProperties, Fragment, useEffect, useState} from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';
import {ThreeDots, MutatingDots} from 'react-loader-spinner'
import {useNavigate} from 'react-router-dom';
import {FaChevronLeft} from "react-icons/fa6";
import {TiDocumentText} from "react-icons/ti";


const CodeReview = () => {
  const navigate = useNavigate();
  const [isHovering, setIsHovering] = useState<boolean>(false)
  const [containerSize, setContainerSize] = useState('w-250 h-200')
  const [rating, setRating] = useState<number>(0)
  const [ratings, setRatings] = useState<number[]>([])
  const [strengths, setStrengths] = useState<string[]>(["Extremely up to date on packages and requirements are well documented.", "Git logs are stable and without many conflicts."])
  const [weaknesses, setWeaknesses] = useState<string[]>(["Not well commented or documented in functions or classes.", "Nested for loops and slow implementations."])

  useEffect(() => {
    setContainerSize('w-400 h-300');
  }, []);

  const goBack = () => {
    navigate('/')
  }

  return (
      <div
          className={`flex flex-col items-center ${containerSize} bg-gray-50 rounded-3xl transition-all duration-500 ease-in-out`}>
        <div className={'flex flex-row w-full h-20'}>
          <div className={'flex flex-row items-center w-full max-w-xs bg-transparent pl-6 pt-4'}>
            <button onClick={goBack} className={'mr-2 h-5 w-5'}>
              <FaChevronLeft/>
            </button>
            <h1 className={'font-bold text-4xl mr-3 text-purple-500'}>Score:</h1>
            <button
                onMouseEnter={() => setIsHovering(true)}
                onMouseLeave={() => setIsHovering(false)}
            >
              <AnimatedTextGradient className={"font-black text-5xl"} text={'8/10'}></AnimatedTextGradient>
            </button>
          </div>
          <div className={'flex w-full justify-end items-center mt-3 mr-4'}>
            <button className={'h-8 w-8'}>
              <TiDocumentText className={'h-8 w-8'}/>
            </button>
          </div>
        </div>
        <div className='flex flex-1 w-full'>
          <div
              className='flex flex-1 border-4 border-purple-500 my-2 ml-3 r-1 p-2 rounded-2xl'>  {/* Flex-1 for each to fill available space horizontally */}
            <div className={'flex flex-col w-full h-full'}>
              <h1 className={'font-semibold text-xl text-black'}>Strengths</h1>
              <ul className="list-disc pl-5">
                {strengths.map((item, index) => (
                    <li key={index} className="font-medium text-sm text-black">
                      {item}
                    </li>
                ))}
              </ul>
            </div>
          </div>
          <div
              className='flex flex-1 border-4 border-purple-500 my-2 mr-3 ml-1 p-2 rounded-2xl'>  {/* Flex-1 for each to fill available space horizontally */}
            <div className={'flex flex-col w-full h-full'}>
              <h1 className={'font-semibold text-xl text-black'}>Weaknesses</h1>
              <ul className="list-disc pl-5">
                {weaknesses.map((item, index) => (
                    <li key={index} className="font-medium text-sm text-black">
                      {item}
                    </li>))}
              </ul>
            </div>
          </div>
        </div>
      </div>
  );
};

export default CodeReview;
