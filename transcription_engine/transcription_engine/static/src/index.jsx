// File: transcription_engine/static/src/index.jsx
import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import TranscriptTimeline from '@/components/TranscriptTimeline';
import './styles/main.css';

function App() {
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/transcript')
      .then(response => response.json())
      .then(data => {
        setSegments(data);
        setLoading(false);
      })
      .catch(err => {
        setError('Failed to load transcript');
        setLoading(false);
        console.error('Error fetching transcript:', err);
      });
  }, []);

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500">{error}</div>;
  }

  return (
    <div className="p-4">
      <TranscriptTimeline segments={segments} />
    </div>
  );
}

const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
