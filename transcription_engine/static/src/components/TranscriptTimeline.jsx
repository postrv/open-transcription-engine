// File: transcription_engine/static/src/components/TranscriptTimeline.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import TranscriptSegment from './TranscriptSegment';
import WaveformPlayer from './WaveformPlayer';
import SpeakerManager from './SpeakerManager';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Download, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useInfiniteScroll } from '../hooks/useInfiniteScroll';

const SEGMENTS_PER_PAGE = 15;
const POLL_INTERVAL = 2000;

const TranscriptTimeline = ({
  segments = [],
  onUpdate,
  audioUrl,
  jobId
}) => {
  const [localSegments, setLocalSegments] = useState(segments);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const unmountingRef = useRef(false);
  const pollIntervalRef = useRef(null);

  const {
    visibleItems: visibleSegments,
    loading,
    hasMore,
    loaderRef,
  } = useInfiniteScroll(localSegments, SEGMENTS_PER_PAGE);

  const fetchTranscript = useCallback(async () => {
    if (!jobId || unmountingRef.current) {
      console.log('Skipping transcript fetch - no jobId or unmounting');
      return;
    }

    console.log(`Fetching transcript for job ${jobId}`);
    try {
      const response = await fetch(`/api/transcript/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch transcript: ${response.statusText}`);
      }

      const rawData = await response.json();
      console.log('Received transcript data:', rawData);

      if (!unmountingRef.current) {
        // Handle both array and object response formats
        const data = Array.isArray(rawData) ? rawData : rawData.segments || [];

        if (data.length > 0) {
          console.log(`Setting ${data.length} segments to local state`);
          setLocalSegments(data);
          onUpdate?.(data); // Notify parent of update
          setError(null);

          // Stop polling if we have complete segments
          const lastSegment = data[data.length - 1];
          if (lastSegment && lastSegment.end > 0) {
            console.log('Transcript complete, stopping polling');
            setIsPolling(false);
          }
        } else {
          console.log('No segments in response');
        }
      }
    } catch (error) {
      console.error('Error fetching transcript:', error);
      if (!unmountingRef.current) {
        setError(`Failed to load transcript: ${error.message}`);
      }
    } finally {
      setIsInitialLoad(false);
    }
  }, [jobId, onUpdate]);

  // Effect for polling setup
  useEffect(() => {
    if (!jobId) {
      console.log('No jobId available, skipping poll setup');
      return;
    }

    unmountingRef.current = false;
    setIsPolling(true);
    setIsInitialLoad(true);

    console.log(`Setting up polling for job ${jobId}`);

    // Initial fetch
    fetchTranscript();

    // Set up polling interval
    pollIntervalRef.current = setInterval(() => {
      if (isPolling && !unmountingRef.current) {
        fetchTranscript();
      }
    }, POLL_INTERVAL);

    // Cleanup
    return () => {
      console.log('Cleaning up polling');
      unmountingRef.current = true;
      setIsPolling(false);
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [jobId, fetchTranscript, isPolling]);

  // Update segments from props
  useEffect(() => {
    console.log('TranscriptTimeline received segments:', segments);
    if (segments?.length > 0) {
      setLocalSegments(segments);
    }
  }, [segments]);

  // Update duration when segments change
  useEffect(() => {
    if (localSegments?.length > 0) {
      const lastSegment = localSegments[localSegments.length - 1];
      setDuration(lastSegment.end);
    }
  }, [localSegments]);

  const handleSegmentUpdate = useCallback((index, updatedSeg) => {
    setLocalSegments(prev => {
      if (!Array.isArray(prev)) return prev;
      const newList = [...prev];

      // Handle split segments (when updatedSeg is an array)
      if (Array.isArray(updatedSeg)) {
        // Remove the original segment
        newList.splice(index, 1);
        // Insert the new segments in its place
        newList.splice(index, 0, ...updatedSeg);
      } else {
        // Regular single segment update
        newList[index] = updatedSeg;
      }

      // Re-sort segments by start time to maintain order
      const sortedList = newList.sort((a, b) => a.start - b.start);

      // Notify parent of update
      onUpdate?.(sortedList);
      return sortedList;
    });
  }, [onUpdate]);

  const handleTimeUpdate = useCallback((time) => {
    setCurrentTime(time);
    const currentSegment = localSegments?.find(
      seg => time >= seg.start && time <= seg.end
    );
    if (currentSegment) {
      const element = document.getElementById(`segment-${currentSegment.start}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [localSegments]);

  const handleDownload = useCallback(() => {
    if (!localSegments?.length) return;

    const dataStr = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(localSegments, null, 2)
    )}`;
    const anchor = document.createElement('a');
    anchor.setAttribute('href', dataStr);
    anchor.setAttribute('download', `transcript-${jobId || 'export'}.json`);
    anchor.click();
  }, [localSegments, jobId]);

  // Show initial loading state
  if (isInitialLoad && !localSegments?.length) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            <span>Loading transcript...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-4xl mx-auto px-4 space-y-8">
        {error && (
          <div className="p-4 rounded-lg bg-destructive/10 text-destructive border border-destructive/20">
            {error}
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchTranscript}
              className="ml-2"
            >
              <Loader2 className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        )}

        <Card className="sticky top-4 z-10 shadow-md">
          <CardContent className="p-6">
            <div className="flex justify-between items-center mb-4">
              <div className="text-sm font-medium text-muted-foreground">
                {localSegments?.length || 0} segments
                {isPolling && (
                  <span className="ml-2 text-xs text-muted-foreground">
                    (Updating...)
                  </span>
                )}
              </div>
              <div className="flex items-center space-x-4">
                <Button
                  onClick={handleDownload}
                  variant="outline"
                  size="sm"
                  disabled={!localSegments?.length}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Download JSON
                </Button>
              </div>
            </div>

            {audioUrl && (
              <WaveformPlayer
                audioUrl={audioUrl}
                onTimeUpdate={handleTimeUpdate}
                currentTime={currentTime}
                duration={duration}
              />
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          {visibleSegments?.map((segment, index) => (
            <div
              key={`${segment.start}-${segment.end}`}
              id={`segment-${segment.start}`}
              className={cn(
                "transition-all duration-200",
                currentTime >= segment.start && currentTime <= segment.end
                  ? "scale-[1.02]"
                  : ""
              )}
            >
              <TranscriptSegment
                index={index}
                segment={segment}
                onSegmentUpdate={handleSegmentUpdate}
              />
            </div>
          ))}

          {(loading || hasMore || isPolling) && (
            <div
              ref={loaderRef}
              className="flex justify-center py-8"
            >
              {loading ? (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading more segments...</span>
                </div>
              ) : isPolling ? (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Updating transcript...</span>
                </div>
              ) : (
                <div className="h-8" />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TranscriptTimeline;