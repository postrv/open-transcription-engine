// File: transcription_engine/static/src/index.jsx
import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import TranscriptTimeline from './components/TranscriptTimeline.jsx';
import './styles/main.css';

function App() {
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  useEffect(() => {
    const fetchTranscript = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/transcript');
        if (!response.ok) {
          throw new Error('Failed to load transcript');
        }
        const data = await response.json();
        if (Array.isArray(data)) {
          setSegments(data);
        } else {
          console.warn('Received non-array transcript data:', data);
          setSegments([]);
        }
      } catch (err) {
        console.error('Error fetching transcript:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTranscript();
  }, []);

  const handleAudioUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const response = await fetch('/api/upload-audio', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Failed to upload audio file');
      }
      const data = await response.json();
      setAudioUrl(data.url);
    } catch (err) {
      console.error('Error uploading audio:', err);
      setError('Failed to upload audio file');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 max-w-xl mx-auto mt-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="mb-6">
        <input
          type="file"
          accept="audio/*"
          onChange={handleAudioUpload}
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      </div>

      {segments.length === 0 ? (
        <div className="text-center text-gray-500 mt-8">
          No transcript segments found. Process an audio file to begin.
        </div>
      ) : (
        <TranscriptTimeline segments={segments} audioUrl={audioUrl} />
      )}
    </div>
  );
}

const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
