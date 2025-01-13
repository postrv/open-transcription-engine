import React, { useEffect, useState, useCallback } from 'react';
import { AlertCircle, CheckCircle2, Loader2, RefreshCcw } from 'lucide-react';
import { Progress } from './ui/progress';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';

const MAX_RETRIES = 3;
const RETRY_DELAY = 2000;

const ProcessingStatus = ({ jobId, onComplete }) => {
  const [status, setStatus] = useState('connecting');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [outputPath, setOutputPath] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const [ws, setWs] = useState(null);

  const connect = useCallback(() => {
    if (!jobId) return;

    const socket = new WebSocket(`ws://${window.location.host}/ws/jobs/${jobId}`);

    socket.onopen = () => {
      setStatus('connected');
      setError(null);
      setRetryCount(0);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'processing_update') {
          setStatus(data.data.status);
          setProgress(data.data.progress);

          if (data.data.error) {
            setError(data.data.error);
          }

          if (data.data.output_path) {
            setOutputPath(data.data.output_path);
            onComplete?.(data.data.output_path);
          }
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
        setError('Invalid server response');
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error');
      setStatus('error');
    };

    socket.onclose = () => {
      setStatus('disconnected');

      // Attempt retry if not completed/failed
      if (status !== 'completed' && status !== 'failed' && retryCount < MAX_RETRIES) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          connect();
        }, RETRY_DELAY);
      }
    };

    setWs(socket);

    return () => {
      socket.close();
    };
  }, [jobId, status, retryCount, onComplete]);

  useEffect(() => {
    const cleanup = connect();
    return () => cleanup?.();
  }, [connect]);

  const handleRetry = () => {
    setRetryCount(0);
    setError(null);
    connect();
  };

  const getStatusDisplay = () => {
    switch (status) {
      case 'connecting':
        return 'Connecting...';
      case 'connected':
        return 'Connected, waiting for updates...';
      case 'disconnected':
        return 'Disconnected';
      case 'queued':
        return 'Waiting to process...';
      case 'loading':
        return 'Loading audio file...';
      case 'transcribing':
        return 'Transcribing audio...';
      case 'identifying_speakers':
        return 'Identifying speakers...';
      case 'completed':
        return 'Processing complete!';
      case 'failed':
        return 'Processing failed';
      case 'error':
        return 'Connection error';
      default:
        return status;
    }
  };

  if (!jobId) return null;

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            {status === 'completed' ? (
              <CheckCircle2 className="h-5 w-5 text-green-500" />
            ) : status === 'failed' || status === 'error' ? (
              <AlertCircle className="h-5 w-5 text-destructive" />
            ) : (
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
            )}

            <div className="flex-1">
              <div className="font-medium">{getStatusDisplay()}</div>
              {error && (
                <div className="text-sm text-destructive mt-1">
                  {error}
                  {retryCount > 0 && (
                    <span className="text-muted-foreground ml-2">
                      (Attempt {retryCount}/{MAX_RETRIES})
                    </span>
                  )}
                </div>
              )}
            </div>

            {(status === 'error' || status === 'disconnected') && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleRetry}
                className="gap-2"
              >
                <RefreshCcw className="h-4 w-4" />
                Retry
              </Button>
            )}

            {progress > 0 && progress < 100 && (
              <div className="text-sm text-muted-foreground">
                {Math.round(progress)}%
              </div>
            )}
          </div>

          {progress > 0 && (
            <Progress
              value={progress}
              className={cn(
                status === 'completed' && "bg-green-500",
                status === 'failed' && "bg-destructive"
              )}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ProcessingStatus;
