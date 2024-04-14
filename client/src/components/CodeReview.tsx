/* global chrome */
import React, {CSSProperties, Fragment, useEffect, useState} from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';
import {ThreeDots, MutatingDots} from 'react-loader-spinner'
import {useNavigate} from 'react-router-dom';
import {FaChevronLeft} from "react-icons/fa6";
import {TiDocumentText} from "react-icons/ti";


const CodeReview = () => {
  const navigate = useNavigate();
  const [isHovering, setIsHovering] = useState<boolean[]>(Array.from({length: 10}, (_, i) => false))
  const [containerSize, setContainerSize] = useState('w-250 h-200')
  const [rating, setRating] = useState<number>(0)
  const [ratings, setRatings] = useState<number[]>([7, 3, 2, 9, 5, 7])
  const [strengths, setStrengths] = useState<string[]>(["Extremely up to date on packages and requirements are well documented.", "Git logs are stable and without many conflicts."])
  const [weaknesses, setWeaknesses] = useState<string[]>(["Not well commented or documented in functions or classes.", "Nested for loops and slow implementations."])

  const metrics = ['Functionality', 'Correctness', 'Code Quality', 'Performance', 'Maintainability', 'Usability']
  const ratingColors = [
    {bg: 'purple-50', text: 'black'},
    {bg: 'purple-100', text: 'black'},
    {bg: 'purple-200', text: 'black'},
    {bg: 'purple-300', text: 'black'},
    {bg: 'purple-400', text: 'white'},
    {bg: 'purple-500', text: 'white'},
    {bg: 'purple-500', text: 'white'},
    {bg: 'purple-600', text: 'white'},
    {bg: 'purple-700', text: 'white'},
    {bg: 'purple-800', text: 'white'},
    {bg: 'purple-900', text: 'white'},
    {bg: 'purple-950', text: 'white'},
  ]
  useEffect(() => {
    setContainerSize('w-520 h-400');
  }, []);

  useEffect(() => {

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

            <AnimatedTextGradient className={"font-black text-5xl"} text={'8/10'}></AnimatedTextGradient>
          </div>
          <div className={'flex w-full justify-end items-center mt-3 mr-4'}>
            <button className={'h-8 w-8'}>
              <TiDocumentText className={'h-8 w-8'}/>
            </button>
          </div>
        </div>
        <div className={'flex flex-row justify-evenly items-center mt-1 w-full h-10'}>
          {metrics.map((metric, index) => {
            const colorObj = ratingColors[ratings[index] - 1];
            const color = `bg-${colorObj.bg}`
            const text = `text-${colorObj.text}`
            const textWidth = metrics[index].length * 2
            return (
                <button
                    onMouseEnter={() => {
                      let newHoveringArray = [...isHovering]
                      newHoveringArray[index] = true
                      setIsHovering(newHoveringArray)
                    }}
                    onMouseLeave={() => {
                      let newHoveringArray = [...isHovering]
                      newHoveringArray[index] = false
                      setIsHovering(newHoveringArray)
                    }}
                >
                  {!isHovering[index] ? <div
                      className={`flex justify-center items-center w-12 h-7 ${color} rounded-md`}
                  >
                    <h1 className={`font-semibold ${text} text-lg`}>{ratings[index]}</h1>
                  </div> : <div
                      className={`flex justify-center items-center px-2 w-${textWidth} h-7 ${color} rounded-md`}
                  >
                    <h1 className={`font-medium ${text}`}>{metrics[index]}</h1>
                  </div>}
                </button>
            )
          })}
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
