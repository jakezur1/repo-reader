import React, { useState } from 'react';
import Home from './components/Home';

const App = () => {
  const [page, setPage] = useState('home');

  const navigate = (nextPage) => {
    setPage(nextPage);
  };

  switch (page) {
    case 'home':
    default:
      return <Home navigate={navigate} />;
  }
};

export default App;
