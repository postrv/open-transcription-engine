// File: transcription_engine/static/src/components/App.jsx
import React, { useState, useEffect } from 'react';
import TranscriptTimeline from './TranscriptTimeline';
import { ThemeProvider } from './ThemeProvider';
import AudioUpload from './AudioUpload';
import { Card, CardContent } from './ui/card';
import { MoonIcon, SunIcon, Loader2 } from 'lucide-react';
import { Button } from './ui/button';
import '../styles/main.css';

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

  const handleUploadComplete = (url) => {
    setAudioUrl(url);
  };

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="flex flex-col items-center gap-4 text-muted-foreground">
          <Loader2 className="h-8 w-8 animate-spin" />
          <p>Loading transcript...</p>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider defaultTheme="light" storageKey="vite-ui-theme">
      <div className={`min-h-screen bg-background ${theme}`}>
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container flex h-16 items-center justify-between">
            <div className="flex gap-6 md:gap-10">
              <h1 className="text-xl font-semibold">Transcription Engine</h1>
            </div>
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                className="h-9 w-9"
              >
                {theme === 'light' ? (
                  <SunIcon className="h-4 w-4 rotate-0 scale-100 transition-transform dark:rotate-90 dark:scale-0" />
                ) : (
                  <MoonIcon className="absolute h-4 w-4 rotate-90 scale-0 transition-transform dark:rotate-0 dark:scale-100" />
                )}
                <span className="sr-only">Toggle theme</span>
              </Button>
            </div>
          </div>
        </header>

        <main className="container py-6">
          {error ? (
            <Card className="border-destructive">
              <CardContent className="p-6">
                <div className="text-destructive">{error}</div>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="mb-6">
                <Card>
                  <CardContent className="p-6">
                    <div className="flex flex-col gap-4">
                      <h2 className="text-lg font-semibold">Upload Audio</h2>
                      <AudioUpload onUploadComplete={handleUploadComplete} />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {segments.length === 0 ? (
                <Card>
                  <CardContent className="p-6">
                    <div className="flex flex-col items-center justify-center gap-4 py-8 text-muted-foreground">
                      <p>No transcript segments found. Upload an audio file to begin.</p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <TranscriptTimeline
                  segments={segments}
                  audioUrl={audioUrl}
                  onUpdate={setSegments}
                />
              )}
            </>
          )}
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;
