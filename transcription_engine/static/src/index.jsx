import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import TranscriptTimeline from './components/TranscriptTimeline';
import { ThemeProvider } from './components/ThemeProvider';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { MoonIcon, SunIcon } from 'lucide-react';
import './styles/main.css';

function App() {
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [theme, setTheme] = useState('light');

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

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
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
    <ThemeProvider defaultTheme="light" storageKey="vite-ui-theme">
      <div className={`min-h-screen bg-background text-foreground ${theme}`}>
        <header className="border-b">
          <div className="container mx-auto px-4 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold">Transcription Engine</h1>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
            >
              {theme === 'light' ? (
                <MoonIcon className="h-[1.2rem] w-[1.2rem]" />
              ) : (
                <SunIcon className="h-[1.2rem] w-[1.2rem]" />
              )}
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </header>
        <main className="container mx-auto px-4 py-8">
          <div className="mb-6">
            <Input
              type="file"
              accept="audio/*"
              onChange={handleAudioUpload}
              className="cursor-pointer"
            />
          </div>

          {segments.length === 0 ? (
            <div className="text-center text-muted-foreground mt-8">
              No transcript segments found. Process an audio file to begin.
            </div>
          ) : (
            <TranscriptTimeline segments={segments} audioUrl={audioUrl} />
          )}
        </main>
      </div>
    </ThemeProvider>
  );
}

const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
