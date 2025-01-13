// File: transcription_engine/static/src/components/ProcessingStatus.jsx
import React, { useEffect, useState, useCallback } from 'react';
import { AlertCircle, CheckCircle2, Loader2, RefreshCcw } from 'lucide-react';
import { Progress } from './ui/progress';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { cn } from '@/lib/utils';

const MAX_RETRIES = 3;
const RETRY_DELAY = 2000;

const ProcessingStatus = ({ jobId, onComplete }) => {
  // Start with 'initial' status to prevent premature reconnections
  const [status, setStatus] = useState('initial');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [outputPath, setOutputPath] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const [ws, setWs] = useState(null);
  const [shouldConnect, setShouldConnect] = useState(true);

  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/jobs/${jobId}`;
  }, [jobId]);

  const connect = useCallback(() => {
    if (!jobId || !shouldConnect) return () => {};

    // Close existing connection if any
    if (ws) {
      ws.close();
    }

    const socket = new WebSocket(getWebSocketUrl());
    let reconnectTimeout;

    socket.onopen = () => {
      console.log('WebSocket connected');
      setStatus('connected');
      setError(null);
      setRetryCount(0); // Reset retry count on successful connection
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received message:', data);

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

          // Stop reconnection attempts if processing is complete or failed
          if (['completed', 'failed'].includes(data.data.status)) {
            setShouldConnect(false);
            socket.close(1000, 'Processing finished');
          }
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
        setError('Invalid server response');
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      // Don't set error immediately, let the onclose handler manage reconnection
      setStatus('error');

      // Only show error if we're not attempting to reconnect
      if (retryCount >= MAX_RETRIES) {
        setError('Connection failed after multiple attempts. Please reload the page.');
      }
    };

    socket.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);

      // Don't reconnect for completed jobs or clean closes
      if (event.reason === 'Job already completed' || status === 'completed' || status === 'failed') {
        setShouldConnect(false);
        return;
      }

      // Don't try to reconnect if we closed intentionally
      if (!event.wasClean && shouldConnect && retryCount < MAX_RETRIES) {
        setStatus('disconnected');
        const nextRetry = retryCount + 1;
        setRetryCount(nextRetry);

        if (nextRetry >= MAX_RETRIES) {
          setShouldConnect(false);
          setError('Max retry attempts reached');
          return;
        }

        // Set up reconnection with exponential backoff
        const delay = RETRY_DELAY * Math.pow(2, retryCount);
        reconnectTimeout = setTimeout(() => {
          connect();
        }, delay);
      }
    };

    setWs(socket);

    // Cleanup function
    return () => {
      clearTimeout(reconnectTimeout);
      if (socket.readyState === WebSocket.OPEN) {
        socket.close(1000, 'Component unmounting');
      }
    };
  }, [jobId, retryCount, ws, getWebSocketUrl, onComplete, shouldConnect]);

  useEffect(() => {
    const cleanup = connect();
    return () => cleanup();
  }, [connect]);

  const handleRetry = () => {
    setRetryCount(0);
    setError(null);
    setShouldConnect(true);
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
                  {retryCount > 0 && shouldConnect && (
                    <span className="text-muted-foreground ml-2">
                      (Attempt {retryCount}/{MAX_RETRIES})
                    </span>
                  )}
                </div>
              )}
            </div>

            {(status === 'error' || status === 'disconnected') && shouldConnect && (
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
