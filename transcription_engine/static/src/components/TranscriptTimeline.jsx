import React, { useState, useEffect } from 'react';
import TranscriptSegment from './TranscriptSegment';
import WaveformPlayer from './WaveformPlayer';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Download } from 'lucide-react';

const TranscriptTimeline = ({
  segments = [],
  audioUrl,
  onUpdate,
}) => {
  const [localSegments, setLocalSegments] = useState(segments || []);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

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
    <div className="space-y-6 w-full max-w-4xl mx-auto">
      {audioUrl && (
        <Card className="sticky top-4 z-10 bg-background shadow-md">
          <CardContent className="p-4">
            <WaveformPlayer
              audioUrl={audioUrl}
              onTimeUpdate={handleTimeUpdate}
              currentTime={currentTime}
              duration={duration}
            />
          </CardContent>
        </Card>
      )}

      <div className="flex justify-between items-center">
        <div className="text-sm text-muted-foreground">
          {localSegments.length} segments
        </div>
        <Button onClick={handleDownload} variant="outline" size="sm">
          <Download className="mr-2 h-4 w-4" />
          Download JSON
        </Button>
      </div>

      <div className="space-y-4">
        {localSegments.map((segment, index) => (
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
      </div>
    </div>
  );
};

export default TranscriptTimeline;
