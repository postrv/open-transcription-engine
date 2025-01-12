// File: transcription_engine/static/src/components/TranscriptTimeline.jsx

import React, { useState, useEffect } from 'react';
import TranscriptSegment from './TranscriptSegment';
import WaveformPlayer from './WaveformPlayer';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Download, Loader2 } from 'lucide-react';
import { useInfiniteScroll } from '../hooks/useInfiniteScroll';

const SEGMENTS_PER_PAGE = 15;

const TranscriptTimeline = ({
  segments = [],
  onUpdate,
  audioUrl,
}) => {
  const [localSegments, setLocalSegments] = useState(segments || []);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState(null);

  const {
    visibleItems: visibleSegments,
    loading,
    hasMore,
    loaderRef,
  } = useInfiniteScroll(localSegments, SEGMENTS_PER_PAGE);

  useEffect(() => {
    setLocalSegments(segments || []);
  }, [segments]);

  useEffect(() => {
    if (localSegments.length > 0) {
      const lastSegment = localSegments[localSegments.length - 1];
      setDuration(lastSegment.end);
    }
  }, [localSegments]);

  const handleSegmentUpdate = (index, updatedSeg) => {
    if (!Array.isArray(localSegments)) return;
    const newList = [...localSegments];
    newList[index] = updatedSeg;
    setLocalSegments(newList);
    onUpdate?.(newList);
  };

  const handleTimeUpdate = (time) => {
    setCurrentTime(time);
    const currentSegment = localSegments.find(
      seg => time >= seg.start && time <= seg.end
    );
    if (currentSegment) {
      const element = document.getElementById(`segment-${currentSegment.start}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  };

  const handleDownload = () => {
    const dataStr = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(localSegments, null, 2)
    )}`;
    const anchor = document.createElement('a');
    anchor.setAttribute('href', dataStr);
    anchor.setAttribute('download', 'transcript.json');
    anchor.click();
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-4xl mx-auto px-4 space-y-8">
        {error && (
          <div className="p-4 rounded-lg bg-destructive/10 text-destructive border border-destructive/20">
            {error}
          </div>
        )}

        <Card className="sticky top-4 z-10 shadow-md">
          <CardContent className="p-6">
            <div className="flex justify-between items-center mb-4">
              <div className="text-sm font-medium text-muted-foreground">
                {localSegments.length} segments
              </div>
              <div className="flex items-center space-x-4">
                <Button onClick={handleDownload} variant="outline" size="sm">
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
          {visibleSegments.map((segment, index) => (
            <div
              key={segment.start}
              id={`segment-${segment.start}`}
              className={`transition-all duration-200 ${
                currentTime >= segment.start && currentTime <= segment.end
                  ? 'scale-[1.02]'
                  : ''
              }`}
            >
              <TranscriptSegment
                index={index}
                segment={segment}
                onSegmentUpdate={handleSegmentUpdate}
              />
            </div>
          ))}

          {/* Loading indicator */}
          {(loading || hasMore) && (
            <div
              ref={loaderRef}
              className="flex justify-center py-8"
            >
              {loading ? (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading more segments...</span>
                </div>
              ) : (
                <div className="h-8" /> // Spacer for intersection observer
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TranscriptTimeline;
