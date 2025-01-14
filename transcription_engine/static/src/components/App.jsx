// File: transcription_engine/static/src/components/App.jsx
import React, { useState, useEffect, useCallback, Component } from 'react';
import TranscriptTimeline from './TranscriptTimeline';
import { ThemeProvider, useTheme } from './ThemeProvider';
import AudioUpload from './AudioUpload';
import { Card, CardContent } from './ui/card';
import { MoonIcon, SunIcon, Loader2 } from 'lucide-react';
import { Button } from './ui/button';
import '../styles/main.css';

function AppContent() {
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    const fetchTranscript = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/transcript');
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || 'Failed to load transcript');
        }
        const data = await response.json();
        setSegments(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error('Error fetching transcript:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTranscript();
  }, []);

  const [jobId, setJobId] = useState(null);

  const handleUploadComplete = useCallback((url, newJobId, transcriptData) => {
  console.log('Upload complete:', { url, jobId: newJobId, transcriptData });
  setAudioUrl(url);
  setJobId(newJobId);
  if (transcriptData && Array.isArray(transcriptData)) {
    setSegments(transcriptData);
  }
}, [setSegments]);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
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
    <div className={`min-h-screen bg-background ${theme}`} data-theme={theme}>
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
                <SunIcon className="h-4 w-4" />
              ) : (
                <MoonIcon className="h-4 w-4" />
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
                jobId={jobId}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}

// Wrap the entire app in error boundaries and theme provider
function App() {
  return (
    <ErrorBoundary fallback={<div>Something went wrong</div>}>
      <ThemeProvider defaultTheme="light" storageKey="vite-ui-theme">
        <AppContent />
      </ThemeProvider>
    </ErrorBoundary>
  );
}

// Simple error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

export default App;
