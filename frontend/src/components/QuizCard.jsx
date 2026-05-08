import React, { useState } from 'react';
import MCQCard from './MCQCard';

export default function QuizCard({ questions, onReset }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({}); // { questionIndex: isCorrect }
  const [isFinished, setIsFinished] = useState(false);

  if (!questions || questions.length === 0) {
    return null;
  }

  const handleNext = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      setIsFinished(true);
    }
  };

  const handlePrev = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const handleAnswer = (isCorrect) => {
    setAnswers(prev => ({
      ...prev,
      [currentIndex]: isCorrect
    }));
  };

  if (isFinished) {
    const totalQuestions = questions.length;
    const score = Object.values(answers).filter(Boolean).length;
    const percentage = Math.round((score / totalQuestions) * 100);

    return (
      <div style={styles.resultsContainer}>
        <h2 style={styles.resultsHeading}>Quiz Complete!</h2>
        <div style={styles.scoreCircle}>
          <span style={styles.scoreText}>{score}</span>
          <span style={styles.scoreDivider}>/</span>
          <span style={styles.scoreTotal}>{totalQuestions}</span>
        </div>
        <p style={styles.percentageText}>You scored {percentage}%</p>
        <button onClick={onReset} style={styles.resetButton}>
          Generate New Quiz
        </button>
      </div>
    );
  }

  const currentQuestion = questions[currentIndex];

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.progressText}>
          Question {currentIndex + 1} of {questions.length}
        </span>
        <span style={styles.scoreTracker}>
          Score: {Object.values(answers).filter(Boolean).length}
        </span>
      </div>

      <div style={styles.progressBarWrapper}>
        <div 
          style={{
            ...styles.progressBar, 
            width: `${((currentIndex + 1) / questions.length) * 100}%`
          }}
        ></div>
      </div>

      {/* Force re-render of MCQCard when question changes to reset its state */}
      <MCQCard 
        key={currentQuestion.id || currentIndex} 
        question={currentQuestion} 
        onAnswer={handleAnswer} 
      />

      <div style={styles.navigation}>
        <button 
          onClick={handlePrev} 
          disabled={currentIndex === 0}
          style={{...styles.navButton, opacity: currentIndex === 0 ? 0.5 : 1}}
        >
          &larr; Previous
        </button>
        <button 
          onClick={handleNext}
          style={{...styles.navButton, ...styles.nextButton}}
        >
          {currentIndex === questions.length - 1 ? 'Finish Quiz' : 'Next \u2192'}
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '600px',
    margin: '20px auto',
    fontFamily: 'sans-serif',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px',
  },
  progressText: {
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#495057',
  },
  scoreTracker: {
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#6f42c1',
    backgroundColor: '#f3f0ff',
    padding: '4px 12px',
    borderRadius: '20px',
  },
  progressBarWrapper: {
    width: '100%',
    height: '6px',
    backgroundColor: '#e9ecef',
    borderRadius: '3px',
    marginBottom: '20px',
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#6f42c1',
    transition: 'width 0.3s ease',
  },
  navigation: {
    display: 'flex',
    justifyContent: 'space-between',
    marginTop: '20px',
  },
  navButton: {
    padding: '10px 20px',
    backgroundColor: '#6c757d',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontSize: '16px',
    cursor: 'pointer',
    fontWeight: 'bold',
    transition: 'opacity 0.2s',
  },
  nextButton: {
    backgroundColor: '#007bff',
  },
  resultsContainer: {
    maxWidth: '500px',
    margin: '40px auto',
    padding: '40px 20px',
    textAlign: 'center',
    backgroundColor: '#fff',
    borderRadius: '12px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    fontFamily: 'sans-serif',
  },
  resultsHeading: {
    fontSize: '28px',
    color: '#333',
    margin: '0 0 20px 0',
  },
  scoreCircle: {
    width: '120px',
    height: '120px',
    borderRadius: '50%',
    backgroundColor: '#f8f9fa',
    border: '8px solid #6f42c1',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    margin: '0 auto 20px auto',
  },
  scoreText: {
    fontSize: '36px',
    fontWeight: 'bold',
    color: '#6f42c1',
  },
  scoreDivider: {
    fontSize: '24px',
    color: '#adb5bd',
    margin: '0 4px',
  },
  scoreTotal: {
    fontSize: '24px',
    fontWeight: 'bold',
    color: '#495057',
  },
  percentageText: {
    fontSize: '20px',
    color: '#666',
    marginBottom: '30px',
  },
  resetButton: {
    padding: '12px 24px',
    backgroundColor: '#28a745',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    fontSize: '18px',
    fontWeight: 'bold',
    cursor: 'pointer',
  }
};
