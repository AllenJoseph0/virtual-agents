import { useEffect } from 'react';
import { BrowserRouter } from 'react-router-dom';
import Cookies from 'js-cookie';
import VAgents from './components/virtual-agents/VAgents';
import './App.css';

function App() {
  useEffect(() => {
    // Set the cookies with the specified values
    Cookies.set('userid', '1490');
    Cookies.set('name', 'Allen');
    Cookies.set('usertype', 'ADMINAPP');
    Cookies.set('firmid', '5');
  }, []);

  return (
    <BrowserRouter>
      <VAgents />
    </BrowserRouter>
  );
}

export default App;
