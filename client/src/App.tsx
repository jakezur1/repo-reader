import React from 'react';
import Home from "./components/Home";
import CodeReview from './components/CodeReview';
import './App.css';
import {HashRouter as Router, Routes, Route} from 'react-router-dom';

function App() {
  return (
      <Router>
        <Routes>
          <Route path="/" element={<Home/>}/>
          <Route path="/codeReview" element={<CodeReview/>}/>
        </Routes>
      </Router>
  );
}

export default App;
