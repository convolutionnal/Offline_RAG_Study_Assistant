import React, { useState, useRef } from 'react';
import { uploadPDF } from '../api/client';

export default function FileUpload({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('IDLE'); // IDLE | UPLOADING | SUCCESS
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setError('');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
    }
  };

  const handleFileChange = (e) => {
    setError('');
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      validateAndSetFile(selectedFile);
    }
  };

  const validateAndSetFile = (selectedFile) => {
    if (selectedFile.type !== 'application/pdf' && !selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setError('Please upload a valid PDF file.');
      setFile(null);
      return;
    }
    setFile(selectedFile);
    setStatus('IDLE');
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setStatus('UPLOADING');
    setProgress(0);
    setError('');
    
    try {
      const response = await uploadPDF(file, (percentage) => {
        setProgress(percentage);
      });
      
      setStatus('SUCCESS');
      setResult(response);
      if (onUploadSuccess) {
        onUploadSuccess(response.doc_id);
      }
    } catch (err) {
      setStatus('IDLE');
      setError(err.response?.data?.detail || err.message || 'An error occurred during upload.');
    }
  };

  return (
    <div className="file-upload-container" style={styles.container}>
      {status === 'IDLE' && (
        <div 
          style={styles.dropZone}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <div style={styles.icon}>📄</div>
          <p>Drag and drop a PDF file here</p>
          <p style={styles.orText}>or</p>
          <input 
            type="file" 
            accept=".pdf,application/pdf" 
            ref={fileInputRef}
            onChange={handleFileChange}
            style={styles.hiddenInput}
          />
          <button 
            onClick={() => fileInputRef.current.click()}
            style={styles.browseButton}
          >
            Browse Files
          </button>
          
          {file && (
            <div style={styles.fileInfo}>
              <strong>Selected:</strong> {file.name} ({formatFileSize(file.size)})
              <button onClick={handleUpload} style={styles.uploadButton}>
                Upload Document
              </button>
            </div>
          )}
          
          {error && <p style={styles.errorText}>{error}</p>}
        </div>
      )}

      {status === 'UPLOADING' && (
        <div style={styles.progressContainer}>
          <p>Uploading <strong>{file.name}</strong>...</p>
          <div style={styles.progressBarWrapper}>
            <div style={{...styles.progressBar, width: `${progress}%`}}></div>
          </div>
          <p>{progress}%</p>
        </div>
      )}

      {status === 'SUCCESS' && result && (
        <div style={styles.successContainer}>
          <div style={styles.successIcon}>✅</div>
          <h3>Upload Successful!</h3>
          <p><strong>Document ID:</strong> {result.doc_id}</p>
          <p><strong>Chunks Indexed:</strong> {result.chunks}</p>
          <button 
            onClick={() => {
              setFile(null);
              setResult(null);
              setStatus('IDLE');
            }} 
            style={styles.browseButton}
          >
            Upload Another File
          </button>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '500px',
    margin: '0 auto',
    padding: '20px',
    fontFamily: 'sans-serif',
  },
  dropZone: {
    border: '2px dashed #ccc',
    borderRadius: '8px',
    padding: '40px 20px',
    textAlign: 'center',
    backgroundColor: '#fafafa',
    transition: 'border-color 0.3s',
  },
  icon: {
    fontSize: '48px',
    marginBottom: '10px',
  },
  orText: {
    color: '#666',
    margin: '10px 0',
  },
  hiddenInput: {
    display: 'none',
  },
  browseButton: {
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    padding: '10px 20px',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    marginTop: '10px',
  },
  uploadButton: {
    backgroundColor: '#28a745',
    color: 'white',
    border: 'none',
    padding: '10px 20px',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    marginTop: '15px',
    display: 'block',
    width: '100%',
  },
  fileInfo: {
    marginTop: '20px',
    padding: '15px',
    backgroundColor: '#e9ecef',
    borderRadius: '4px',
    wordBreak: 'break-all',
  },
  errorText: {
    color: '#dc3545',
    marginTop: '15px',
    fontWeight: 'bold',
  },
  progressContainer: {
    textAlign: 'center',
    padding: '40px 20px',
    border: '1px solid #ccc',
    borderRadius: '8px',
  },
  progressBarWrapper: {
    width: '100%',
    backgroundColor: '#e9ecef',
    borderRadius: '4px',
    height: '20px',
    overflow: 'hidden',
    margin: '15px 0',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#007bff',
    transition: 'width 0.3s ease',
  },
  successContainer: {
    textAlign: 'center',
    padding: '40px 20px',
    border: '1px solid #28a745',
    borderRadius: '8px',
    backgroundColor: '#f8fff9',
  },
  successIcon: {
    fontSize: '48px',
    marginBottom: '10px',
  }
};
