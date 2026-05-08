import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import QuizControls from '../components/QuizControls';
import LoadingState from '../components/LoadingState';
import QuizCard from '../components/QuizCard';

export default function Home() {
  const [appState, setAppState] = useState('IDLE'); 
  // State machine: IDLE -> READY -> GENERATING -> QUIZ_DISPLAY
  const [docId, setDocId] = useState(null);
  const [questions, setQuestions] = useState([]);

  const handleUploadSuccess = (uploadedDocId) => {
    setDocId(uploadedDocId);
    setAppState('READY');
  };

  const handleGenerateStart = () => {
    setAppState('GENERATING');
  };

  const handleGenerateError = (errorMsg) => {
    setAppState('READY');
  };

  const handleQuizGenerated = (generatedQuestions) => {
    setQuestions(generatedQuestions);
    setAppState('QUIZ_DISPLAY');
  };

  const handleReset = () => {
    setQuestions([]);
    setAppState('READY');
  };

  return (
    <div style={styles.container}>
      {/* File Upload and Quiz Controls are kept mounted but hidden to preserve state */}
      <div style={{ display: (appState === 'IDLE' || appState === 'READY') ? 'block' : 'none' }}>
        <div style={styles.section}>
          <FileUpload onUploadSuccess={handleUploadSuccess} />
        </div>
        <div style={styles.section}>
          <QuizControls 
            docId={docId} 
            onGenerateStart={handleGenerateStart}
            onQuizGenerated={handleQuizGenerated}
            onGenerateError={handleGenerateError}
          />
        </div>
      </div>

      {/* Loading State shown while quiz is being generated */}
      {appState === 'GENERATING' && (
        <div style={styles.section}>
          <LoadingState />
        </div>
      )}

      {/* Quiz interface shown when questions are ready */}
      {appState === 'QUIZ_DISPLAY' && (
        <div style={styles.section}>
          <QuizCard questions={questions} onReset={handleReset} />
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    padding: '20px',
  },
  section: {
    marginBottom: '30px',
  }
};
