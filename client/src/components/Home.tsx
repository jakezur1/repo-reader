import React from 'react';
import AnimatedTextGradient from './AnimatedTextGradient';

const Home = ({navigate}: any) => {
  return (
      <div className={"w-300 h-300 bg-gray-50 rounded border-4 border-purple-700 m-2"}>
        <div className={'bg-transparent p-3'}>
          <AnimatedTextGradient text={"Welcome to RepoReader!"}></AnimatedTextGradient>
        </div>
        <button className={'flex-1 bg-white border-2 border-purple-700'} onClick={() => {}}>Generate ReadMe
        </button>
      </div>
  );
};

export default Home;
