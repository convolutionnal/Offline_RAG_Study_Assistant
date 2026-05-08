import React, { useState } from 'react';

export default function MCQCard({ question, onAnswer }) {
  const [selectedOption, setSelectedOption] = useState(null);
  const [isRevealed, setIsRevealed] = useState(false);

  const handleOptionClick = (key) => {
    if (isRevealed) return;
    setSelectedOption(key);
  };

  const handleShowAnswer = () => {
    if (!selectedOption) return;
    setIsRevealed(true);
    if (onAnswer) {
      onAnswer(selectedOption === question.correct);
    }
  };

  const getOptionStyle = (key) => {
    const baseStyle = { ...styles.optionButton };
    
    if (isRevealed) {
      if (key === question.correct) {
        return { ...baseStyle, ...styles.optionCorrect };
      }
      if (key === selectedOption && key !== question.correct) {
        return { ...baseStyle, ...styles.optionIncorrect };
      }
      // Dim unselected wrong options
      return { ...baseStyle, opacity: 0.6, cursor: 'default' };
    }
    
    if (key === selectedOption) {
      return { ...baseStyle, ...styles.optionSelected };
    }
    
    return baseStyle;
  };

  return (
    <div style={styles.card}>
      <h4 style={styles.questionText}>
        <span style={styles.questionNumber}>Q{question.id}.</span> {question.question}
      </h4>
      
      <div style={styles.optionsContainer}>
        {Object.entries(question.options).map(([key, text]) => (
          <button
            key={key}
            onClick={() => handleOptionClick(key)}
            style={getOptionStyle(key)}
            disabled={isRevealed}
          >
            <span style={styles.optionLabel}>{key}</span> {text}
          </button>
        ))}
      </div>

      {!isRevealed && selectedOption && (
        <button onClick={handleShowAnswer} style={styles.showAnswerButton}>
          Show Answer
        </button>
      )}

      {isRevealed && (
        <div style={styles.explanationContainer}>
          <h5 style={styles.explanationTitle}>Explanation:</h5>
          <p style={styles.explanationText}>{question.explanation}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1)',
    padding: '24px',
    marginBottom: '20px',
    fontFamily: 'sans-serif',
    border: '1px solid #eaeaea',
  },
  questionText: {
    fontSize: '18px',
    lineHeight: '1.5',
    margin: '0 0 20px 0',
    color: '#333',
  },
  questionNumber: {
    color: '#6f42c1',
    fontWeight: 'bold',
  },
  optionsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  optionButton: {
    display: 'flex',
    alignItems: 'flex-start',
    textAlign: 'left',
    padding: '12px 16px',
    backgroundColor: '#f8f9fa',
    border: '2px solid #e9ecef',
    borderRadius: '8px',
    fontSize: '16px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    color: '#333',
    lineHeight: '1.4',
  },
  optionLabel: {
    fontWeight: 'bold',
    marginRight: '12px',
    color: '#6c757d',
    minWidth: '20px',
  },
  optionSelected: {
    borderColor: '#007bff',
    backgroundColor: '#f0f7ff',
  },
  optionCorrect: {
    borderColor: '#28a745',
    backgroundColor: '#f8fff9',
    color: '#155724',
  },
  optionIncorrect: {
    borderColor: '#dc3545',
    backgroundColor: '#fff8f8',
    color: '#721c24',
  },
  showAnswerButton: {
    marginTop: '20px',
    padding: '10px 20px',
    backgroundColor: '#6c757d',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '16px',
    fontWeight: 'bold',
    cursor: 'pointer',
  },
  explanationContainer: {
    marginTop: '20px',
    padding: '16px',
    backgroundColor: '#f8f9fa',
    borderLeft: '4px solid #6f42c1',
    borderRadius: '4px',
  },
  explanationTitle: {
    margin: '0 0 8px 0',
    color: '#495057',
    fontSize: '16px',
  },
  explanationText: {
    margin: '0',
    color: '#333',
    lineHeight: '1.5',
  }
};
