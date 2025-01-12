// File: transcription_engine/static/src/components/TranscriptSegment.jsx
import React, { useState } from "react";
import { Card, CardContent, CardFooter } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Edit, Save, X, Clock, User, AlertTriangle } from 'lucide-react';

const TranscriptSegment = ({
  segment,
  onSegmentUpdate,
  index,
}) => {
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

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'bg-green-500';
    if (confidence >= 0.7) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <Card className={`
      w-full transition-all duration-300
      hover:shadow-md
      ${isEditing ? 'ring-2 ring-primary' : ''}
    `}>
      <CardContent className="p-4">
        <div className="flex flex-col gap-3">
          <div className="flex justify-between items-start">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>{`${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s`}</span>
            </div>

            <div className="flex items-center gap-2">
              {segment.confidence < 0.7 && (
                <div className="flex items-center gap-1 text-yellow-500">
                  <AlertTriangle className="h-4 w-4" />
                  <span className="text-xs">Low confidence</span>
                </div>
              )}

              {isEditing ? (
                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <Input
                    type="text"
                    value={draftSpeaker}
                    onChange={(e) => setDraftSpeaker(e.target.value)}
                    className="w-40 h-8"
                    placeholder="Speaker"
                  />
                </div>
              ) : (
                segment.speaker_id && (
                  <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-secondary">
                    <User className="h-4 w-4" />
                    <span className="text-sm font-medium">{segment.speaker_id}</span>
                  </div>
                )
              )}
            </div>
          </div>

          {isEditing ? (
            <Textarea
              value={draftText}
              onChange={(e) => setDraftText(e.target.value)}
              className="mt-2 min-h-[100px] text-base"
              rows={4}
            />
          ) : (
            <div className="mt-2 text-base leading-relaxed">
              {segment.text}
            </div>
          )}

          {segment.confidence && (
            <div className="flex items-center gap-2 mt-2">
              <div className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getConfidenceColor(segment.confidence)}`}
                  style={{ width: `${segment.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground">
                {(segment.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>
          )}
        </div>
      </CardContent>

      <CardFooter className="px-4 py-2 border-t">
        {!isEditing ? (
          <Button
            onClick={() => setIsEditing(true)}
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground gap-2"
          >
            <Edit className="h-4 w-4" />
            Edit
          </Button>
        ) : (
          <div className="flex gap-2">
            <Button
              onClick={handleSave}
              variant="default"
              size="sm"
              className="gap-2"
            >
              <Save className="h-4 w-4" />
              Save
            </Button>
            <Button
              onClick={handleCancel}
              variant="ghost"
              size="sm"
              className="gap-2 text-destructive hover:bg-destructive/10"
            >
              <X className="h-4 w-4" />
              Cancel
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
};

export default TranscriptSegment;
