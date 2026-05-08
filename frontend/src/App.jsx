import React from 'react';
import Home from './pages/Home';

function App() {
  return (
    <div style={styles.appContainer}>
      <header style={styles.header}>
        <h1 style={styles.title}>Offline RAG Quiz Generator</h1>
      </header>
      <main>
        <Home />
      </main>
    </div>
  );
}

const styles = {
  appContainer: {
    minHeight: '100vh',
    backgroundColor: '#f8f9fa',
    fontFamily: 'sans-serif',
  },
  header: {
    backgroundColor: '#6f42c1',
    color: '#fff',
    padding: '20px',
    textAlign: 'center',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  title: {
    margin: 0,
    fontSize: '28px',
  }
};

export default App;
