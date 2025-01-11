// File: transcription_engine/static/src/components/TranscriptTimeline.jsx
import React, { useState } from 'react';
import TranscriptSegment from './TranscriptSegment';

const TranscriptTimeline = ({
  segments = [],
  onUpdate, // Optional callback from parent (if needed)
}) => {
  const [localSegments, setLocalSegments] = useState(segments);

  const handleSegmentUpdate = (index, updatedSeg) => {
    const newList = [...localSegments];
    newList[index] = updatedSeg;
    setLocalSegments(newList);
    onUpdate?.(newList);
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
      <div className="flex justify-end mb-4">
        <button
          onClick={handleDownload}
          className="text-sm px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Download JSON
        </button>
      </div>
      {localSegments.map((segment, index) => (
        <TranscriptSegment
          key={index}
          index={index}
          segment={segment}
          onSegmentUpdate={handleSegmentUpdate}
        />
      ))}
    </div>
  );
};

export default TranscriptTimeline;
