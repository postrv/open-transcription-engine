// File: transcription_engine/static/src/components/TranscriptTimeline.jsx
import React from 'react';
import { Card } from './ui/card';

const TranscriptTimeline = ({ segments = [], onUpdate }) => {
  return (
    <div className="space-y-4 w-full max-w-4xl mx-auto">
      {segments.map((segment, index) => (
        <Card key={index} className="p-4 bg-white shadow-sm hover:shadow-md transition-shadow">
          <div className="flex justify-between items-start">
            <div className="text-sm text-gray-500">
              {segment.start.toFixed(2)}s - {segment.end.toFixed(2)}s
            </div>
            {segment.speaker_id && (
              <div className="text-sm font-medium text-blue-600">
                {segment.speaker_id}
              </div>
            )}
          </div>
          <div className="mt-2 text-gray-900">{segment.text}</div>
          {segment.confidence && (
            <div className="mt-1 text-xs text-gray-400">
              Confidence: {(segment.confidence * 100).toFixed(1)}%
            </div>
          )}
        </Card>
      ))}
    </div>
  );
};

export default TranscriptTimeline;
