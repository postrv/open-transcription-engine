// File: transcription_engine/static/src/components/TranscriptTimeline.jsx
import React, { useState, useEffect } from 'react';
import TranscriptSegment from './TranscriptSegment.jsx';
import WaveformPlayer from './WaveformPlayer.jsx';

const TranscriptTimeline = ({
  segments = [],
  audioUrl,
  onUpdate, // Optional callback from parent (if needed)
}) => {
  const [localSegments, setLocalSegments] = useState(segments || []);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Update localSegments when segments prop changes
  useEffect(() => {
    setLocalSegments(segments || []);
  }, [segments]);

  // Update duration when segments change
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
    // Find and highlight current segment
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
    // Simple JSON download
    const dataStr = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(localSegments, null, 2)
    )}`;
    const anchor = document.createElement('a');
    anchor.setAttribute('href', dataStr);
    anchor.setAttribute('download', 'transcript.json');
    anchor.click();
  };

  return (
    <div className="space-y-4 w-full max-w-4xl mx-auto">
      {/* Audio player */}
      {audioUrl && (
        <div className="sticky top-0 z-10 bg-white p-4 shadow-md rounded-lg mb-8">
          <WaveformPlayer
            audioUrl={audioUrl}
            onTimeUpdate={handleTimeUpdate}
            currentTime={currentTime}
            duration={duration}
          />
        </div>
      )}

      <div className="flex justify-between mb-4">
        <div className="text-sm text-gray-600">
          {localSegments.length} segments
        </div>
        <button
          onClick={handleDownload}
          className="text-sm px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Download JSON
        </button>
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
