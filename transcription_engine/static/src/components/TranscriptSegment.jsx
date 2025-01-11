// File: transcription_engine/static/src/components/TranscriptSegment.jsx
import React, { useState } from "react";
import { Card } from "./ui/card.jsx";
import { cn } from "../lib/utils.js";

/**
 * Single segment in the transcript timeline
 */
const TranscriptSegment = ({
  segment,
  onSegmentUpdate,
  index,
}) => {
  // Local states for editing
  const [isEditing, setIsEditing] = useState(false);
  const [draftText, setDraftText] = useState(segment.text);
  const [draftSpeaker, setDraftSpeaker] = useState(segment.speaker_id || "");

  const handleSave = () => {
    setIsEditing(false);
    onSegmentUpdate?.(index, {
      ...segment,
      text: draftText,
      speaker_id: draftSpeaker,
    });
  };

  const handleCancel = () => {
    setIsEditing(false);
    setDraftText(segment.text);
    setDraftSpeaker(segment.speaker_id || "");
  };

  return (
    <Card className="p-4 bg-white shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start">
        <div className="text-sm text-gray-500">
          {`${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s`}
        </div>
        {isEditing ? (
          <input
            type="text"
            value={draftSpeaker}
            onChange={(e) => setDraftSpeaker(e.target.value)}
            className={cn(
              "border text-sm px-2 py-1 rounded",
              "focus:outline-none focus:ring-1 focus:ring-blue-300"
            )}
            placeholder="Speaker"
          />
        ) : (
          segment.speaker_id && (
            <div className="text-sm font-medium text-blue-600">
              {segment.speaker_id}
            </div>
          )
        )}
      </div>
      {isEditing ? (
        <textarea
          className={cn(
            "mt-2 text-gray-900 w-full border rounded",
            "focus:outline-none focus:ring-1 focus:ring-blue-300"
          )}
          value={draftText}
          onChange={(e) => setDraftText(e.target.value)}
          rows={4}
        />
      ) : (
        <div className="mt-2 text-gray-900">{segment.text}</div>
      )}
      {segment.confidence && (
        <div className="mt-1 text-xs text-gray-400">
          Confidence: {(segment.confidence * 100).toFixed(1)}%
        </div>
      )}
      {/* Action buttons */}
      <div className="mt-3 flex space-x-2">
        {!isEditing ? (
          <button
            onClick={() => setIsEditing(true)}
            className="text-sm px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            Edit
          </button>
        ) : (
          <>
            <button
              onClick={handleSave}
              className="text-sm px-3 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200"
            >
              Save
            </button>
            <button
              onClick={handleCancel}
              className="text-sm px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
            >
              Cancel
            </button>
          </>
        )}
      </div>
    </Card>
  );
};

export default TranscriptSegment;
