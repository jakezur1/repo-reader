import React from 'react';

const Home = ({ navigate }: any) => {
  return (
      <div className={"w-300 h-300 rounded-2rem"}>
        <div className={'bg-transparent'}>
          <h1 className={'text-4xl m-3 font-bold text-purple-700'}>Welcome to RepoReader</h1>
        </div>
        <button className={'flex-1 bg-white border-2 border-purple-700'} onClick={() => {
        }}>Generate ReadMe</button>
      </div>
  );
};

export default Home;
