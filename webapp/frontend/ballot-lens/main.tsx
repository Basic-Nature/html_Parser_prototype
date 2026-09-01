import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { App } from './app/App';
import { readBallotLensBootstrap } from './contracts/bootstrap';
import './styles/tokens.css';
import './styles/shell.css';

const rootElement = document.getElementById('ballotLensF2Root');

if (!(rootElement instanceof HTMLElement)) {
  throw new Error('Ballot Lens F2 root is missing');
}

const bootstrap = readBallotLensBootstrap(rootElement);

createRoot(rootElement).render(
  <StrictMode>
    <App bootstrap={bootstrap} />
  </StrictMode>,
);
