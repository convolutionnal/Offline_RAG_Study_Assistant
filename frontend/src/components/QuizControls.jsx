import React, { useState } from 'react';
import { generateQuiz } from '../api/client';

export default function QuizControls({ docId, onQuizGenerated, onGenerateStart, onGenerateError }) {
  const [topic, setTopic] = useState('');
  const [numQuestions, setNumQuestions] = useState(5);
  const [difficulty, setDifficulty] = useState('medium');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  if (!docId) {
    return (
      <div style={styles.disabledContainer}>
        <p style={styles.disabledText}>Upload a PDF first to generate a quiz</p>
      </div>
    );
  }

  const handleGenerate = async () => {
    if (!topic.trim()) {
      setError('Please enter a topic.');
      return;
    }

    setIsLoading(true);
    setError('');
    
    if (onGenerateStart) {
      onGenerateStart();
    }

    try {
      const response = await generateQuiz(docId, topic, numQuestions, difficulty);
      if (onQuizGenerated && response.questions) {
        onQuizGenerated(response.questions);
      }
    } catch (err) {
      let errorMsg = err.response?.data?.detail || err.message || 'Failed to generate quiz.';
      if (err.response?.status === 400) {
        errorMsg = "0 chunks match do it again.";
      }
      setError(errorMsg);
      if (onGenerateError) {
        onGenerateError(errorMsg);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.heading}>Configure Quiz</h3>
      
      <div style={styles.inputGroup}>
        <label style={styles.label}>Topic:</label>
        <input 
          type="text" 
          value={topic} 
          onChange={(e) => setTopic(e.target.value)} 
          placeholder="e.g. process scheduling, virtual memory" 
          style={styles.textInput}
          disabled={isLoading}
        />
      </div>

      <div style={styles.inputGroup}>
        <label style={styles.label}>
          Number of questions: <strong>{numQuestions}</strong>
        </label>
        <input 
          type="range" 
          min="1" 
          max="20" 
          value={numQuestions} 
          onChange={(e) => setNumQuestions(parseInt(e.target.value, 10))}
          style={styles.rangeInput}
          disabled={isLoading}
        />
      </div>

      <div style={styles.inputGroup}>
        <label style={styles.label}>Difficulty:</label>
        <select 
          value={difficulty} 
          onChange={(e) => setDifficulty(e.target.value)}
          style={styles.selectInput}
          disabled={isLoading}
        >
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>

      <button 
        onClick={handleGenerate} 
        disabled={isLoading || !topic.trim()} 
        style={{...styles.generateButton, opacity: isLoading || !topic.trim() ? 0.7 : 1}}
      >
        {isLoading ? (
          <span style={styles.spinnerWrapper}>
            <span style={styles.spinner}></span> Generating...
          </span>
        ) : (
          'Generate Quiz'
        )}
      </button>

      {error && <p style={styles.errorText}>{error}</p>}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '500px',
    margin: '20px auto',
    padding: '20px',
    border: '1px solid #ccc',
    borderRadius: '8px',
    fontFamily: 'sans-serif',
    backgroundColor: '#fff',
  },
  disabledContainer: {
    maxWidth: '500px',
    margin: '20px auto',
    padding: '40px 20px',
    border: '1px dashed #ccc',
    borderRadius: '8px',
    fontFamily: 'sans-serif',
    backgroundColor: '#fafafa',
    textAlign: 'center',
  },
  disabledText: {
    color: '#666',
    fontWeight: 'bold',
    fontSize: '18px',
  },
  heading: {
    marginTop: 0,
    marginBottom: '20px',
  },
  inputGroup: {
    marginBottom: '15px',
  },
  label: {
    display: 'block',
    marginBottom: '5px',
    fontWeight: 'bold',
  },
  textInput: {
    width: '100%',
    padding: '10px',
    boxSizing: 'border-box',
    borderRadius: '4px',
    border: '1px solid #ccc',
    fontSize: '16px',
  },
  rangeInput: {
    width: '100%',
    marginTop: '5px',
  },
  selectInput: {
    width: '100%',
    padding: '10px',
    borderRadius: '4px',
    border: '1px solid #ccc',
    fontSize: '16px',
  },
  generateButton: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#6f42c1',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    fontSize: '16px',
    fontWeight: 'bold',
    cursor: 'pointer',
    marginTop: '10px',
    transition: 'opacity 0.2s',
  },
  errorText: {
    color: '#dc3545',
    marginTop: '15px',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  spinnerWrapper: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
  },
  spinner: {
    display: 'inline-block',
    width: '16px',
    height: '16px',
    border: '3px solid rgba(255,255,255,.3)',
    borderRadius: '50%',
    borderTopColor: '#fff',
    animation: 'spin 1s ease-in-out infinite',
  }
};

if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.innerHTML = `
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
}
