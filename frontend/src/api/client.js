import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 480000, // 480s (8 minutes) because LLM and PDF indexing are slow
});

// Response interceptor to log errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error(
        `API Error: [${error.config.method.toUpperCase()} ${error.config.url}] - Status: ${error.response.status}`
      );
    } else {
      console.error(`API Error: ${error.message}`);
    }
    return Promise.reject(error);
  }
);

export const uploadPDF = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onUploadProgress(percentage);
      }
    },
  });
  return response.data;
};

export const generateQuiz = async (docId, topic, numQuestions, difficulty) => {
  console.log('Sending doc_id:', docId);
  const response = await apiClient.post('/quiz', {
    doc_id: docId,
    topic: topic,
    num_questions: numQuestions,
    difficulty: difficulty,
  });
  return response.data;
};

export const checkHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};
