// File: transcription_engine/static/src/components/ProcessingStatus.jsx
import React, { useEffect, useState, useCallback, useRef } from 'react';
import { AlertCircle, CheckCircle2, Loader2, RefreshCcw } from 'lucide-react';
import { Progress } from './ui/progress';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { cn } from '@/lib/utils';

const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY = 2000;
const BACKOFF_FACTOR = 1.5;

const ProcessingStatus = ({ jobId, onComplete }) => {
  const [state, setState] = useState({
    status: 'initial',
    progress: 0,
    error: null,
    outputPath: null,
  });
  const [retryCount, setRetryCount] = useState(0);
  const [shouldConnect, setShouldConnect] = useState(true);
  const wsRef = useRef(null);
  const retryTimeoutRef = useRef(null);
  const unmountingRef = useRef(false);

  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/jobs/${jobId}`;
  }, [jobId]);

  const clearRetryTimeout = useCallback(() => {
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }, []);

  const handleUpdateMessage = useCallback(async (update) => {
    console.log('Processing update:', update);
    setState(prev => ({
      ...prev,
      status: update.status,
      progress: update.progress || prev.progress,
      error: update.error || null,
      outputPath: update.output_path || prev.outputPath
    }));

    if (update.output_path) {
      console.log('Got output path:', update.output_path);
      try {
        // Fetch the transcript data
        const response = await fetch(`/api/transcript/${jobId}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch transcript: ${response.statusText}`);
        }
        const transcriptData = await response.json();
        console.log('Fetched transcript data:', transcriptData);

        // Call onComplete with both the path and the data
        await onComplete?.(update.output_path, transcriptData);
      } catch (error) {
        console.error('Error fetching transcript:', error);
        setState(prev => ({ ...prev, error: error.message }));
      }
    }

    if (['completed', 'failed'].includes(update.status)) {
      console.log('Job finished:', update.status);
      setShouldConnect(false);
    }
  }, [jobId, onComplete]);

  const connect = useCallback(() => {
    if (!jobId || !shouldConnect || unmountingRef.current) {
      return;
    }

    clearRetryTimeout();

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close(1000, 'Reconnecting');
    }

    try {
      console.log('Connecting to WebSocket...');
      const socket = new WebSocket(getWebSocketUrl());
      wsRef.current = socket;

      socket.onopen = () => {
        if (unmountingRef.current) {
          socket.close(1000, 'Component unmounted');
          return;
        }
        console.log('WebSocket connected for job:', jobId);
        setState(prev => ({ ...prev, status: 'connected', error: null }));
        setRetryCount(0);
      };

      socket.onmessage = async (event) => {
        if (unmountingRef.current) return;

        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data);

          if (data.type === 'processing_update') {
            await handleUpdateMessage(data.data);
          }
        } catch (err) {
          console.error('Error processing WebSocket message:', err);
          setState(prev => ({ ...prev, error: 'Invalid server response' }));
        }
      };

      socket.onerror = (error) => {
        if (unmountingRef.current) return;
        console.error('WebSocket error:', error);
        setState(prev => ({ ...prev, status: 'error' }));
      };

      socket.onclose = (event) => {
        if (unmountingRef.current) return;
        console.log('WebSocket closed:', event.code, event.reason);

        if (event.code === 1000 ||
            event.reason === 'Job already completed' ||
            ['completed', 'failed'].includes(state.status) ||
            !shouldConnect) {
          return;
        }

        if (retryCount < MAX_RETRIES) {
          setState(prev => ({ ...prev, status: 'disconnected' }));
          const nextRetry = retryCount + 1;
          setRetryCount(nextRetry);
          const delay = INITIAL_RETRY_DELAY * Math.pow(BACKOFF_FACTOR, retryCount);

          retryTimeoutRef.current = setTimeout(() => {
            if (!unmountingRef.current && shouldConnect) {
              connect();
            }
          }, delay);
        } else {
          setShouldConnect(false);
          setState(prev => ({
            ...prev,
            error: 'Connection attempts exhausted'
          }));
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setState(prev => ({
        ...prev,
        error: `Failed to create WebSocket connection: ${error.message}`
      }));
    }
  }, [jobId, retryCount, getWebSocketUrl, handleUpdateMessage, clearRetryTimeout, shouldConnect, state.status]);

  useEffect(() => {
    console.log('ProcessingStatus: jobId changed:', jobId);
    if (jobId) {
      unmountingRef.current = false;
      connect();
    }

    return () => {
      unmountingRef.current = true;
      clearRetryTimeout();
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close(1000, 'Component unmounting');
      }
    };
  }, [jobId, connect, clearRetryTimeout]);

  const handleRetry = useCallback(() => {
    setRetryCount(0);
    setState(prev => ({ ...prev, error: null }));
    setShouldConnect(true);
    connect();
  }, [connect]);

  const getStatusDisplay = useCallback(() => {
    switch (state.status) {
      case 'initial':
      case 'connecting':
        return 'Connecting to processing service...';
      case 'connected':
        return 'Connected, awaiting updates...';
      case 'disconnected':
        return 'Connection lost - attempting to reconnect...';
      case 'queued':
        return 'Waiting in processing queue...';
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
        return `Status: ${state.status}`;
    }
  }, [state.status]);

  if (!jobId) return null;

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            {state.status === 'completed' ? (
              <CheckCircle2 className="h-5 w-5 text-green-500" />
            ) : state.status === 'failed' || state.status === 'error' ? (
              <AlertCircle className="h-5 w-5 text-destructive" />
            ) : (
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
            )}

            <div className="flex-1">
              <div className="font-medium">{getStatusDisplay()}</div>
              {state.error && (
                <div className="text-sm text-destructive mt-1">
                  {state.error}
                  {retryCount > 0 && shouldConnect && (
                    <span className="text-muted-foreground ml-2">
                      (Attempt {retryCount}/{MAX_RETRIES})
                    </span>
                  )}
                </div>
              )}
            </div>

            {(state.status === 'error' || state.status === 'disconnected') && shouldConnect && (
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

            {state.progress > 0 && state.progress < 100 && (
              <div className="text-sm text-muted-foreground">
                {Math.round(state.progress)}%
              </div>
            )}
          </div>

          {state.progress > 0 && (
            <Progress
              value={state.progress}
              className={cn(
                state.status === 'completed' && "bg-green-500",
                state.status === 'failed' && "bg-destructive"
              )}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ProcessingStatus;
