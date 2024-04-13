import React, {useState} from 'react';
import Home from "./components/Home"
import './App.css';

function App() {
  const [page, setPage] = useState('home');

  const navigate = (nextPage: any) => {
    setPage(nextPage);
  };

  switch (page) {
    case 'home':
    default:
      return <Home navigate={navigate} />;
  }
}

export default App;
