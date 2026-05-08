import React, { useState, useEffect } from 'react';

export default function LoadingState() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(s => s + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.spinner}></div>
        <h3 style={styles.title}>Generating... {seconds}s</h3>
      </div>
      <p style={styles.subtitle}>
        Local LLM is thinking — this takes 20-40 seconds
      </p>

      <div style={styles.skeletonsContainer}>
        {[1, 2, 3].map((i) => (
          <div key={i} style={styles.skeletonCard}>
            <div style={{...styles.skeletonLine, width: '70%', height: '24px', marginBottom: '20px'}}></div>
            <div style={{...styles.skeletonLine, width: '100%', height: '48px', marginBottom: '12px'}}></div>
            <div style={{...styles.skeletonLine, width: '100%', height: '48px', marginBottom: '12px'}}></div>
            <div style={{...styles.skeletonLine, width: '100%', height: '48px', marginBottom: '12px'}}></div>
            <div style={{...styles.skeletonLine, width: '100%', height: '48px'}}></div>
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '600px',
    margin: '40px auto',
    fontFamily: 'sans-serif',
    textAlign: 'center',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '15px',
    marginBottom: '10px',
  },
  spinner: {
    width: '24px',
    height: '24px',
    border: '4px solid #f3f3f3',
    borderTop: '4px solid #6f42c1',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  title: {
    margin: 0,
    fontSize: '24px',
    color: '#333',
  },
  subtitle: {
    margin: '0 0 30px 0',
    color: '#6c757d',
    fontSize: '16px',
    fontWeight: 'bold',
  },
  skeletonsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  skeletonCard: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    padding: '24px',
    border: '1px solid #eaeaea',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.02)',
    textAlign: 'left',
  },
  skeletonLine: {
    backgroundColor: '#eee',
    backgroundImage: 'linear-gradient(90deg, #eee 0px, #f5f5f5 50%, #eee 100%)',
    backgroundSize: '200% 100%',
    borderRadius: '4px',
    animation: 'shimmer 1.5s infinite linear',
  }
};

if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.innerHTML = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
  `;
  document.head.appendChild(style);
}
