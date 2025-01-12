import React, { useState } from "react";
import { Card, CardContent, CardFooter } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Edit, Save, X } from 'lucide-react';

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

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex justify-between items-start mb-2">
          <div className="text-sm text-muted-foreground">
            {`${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s`}
          </div>
          {isEditing ? (
            <Input
              type="text"
              value={draftSpeaker}
              onChange={(e) => setDraftSpeaker(e.target.value)}
              className="w-40"
              placeholder="Speaker"
            />
          ) : (
            segment.speaker_id && (
              <div className="text-sm font-medium text-primary">
                {segment.speaker_id}
              </div>
            )
          )}
        </div>
        {isEditing ? (
          <Textarea
            className="mt-2"
            value={draftText}
            onChange={(e) => setDraftText(e.target.value)}
            rows={4}
          />
        ) : (
          <div className="mt-2">{segment.text}</div>
        )}
        {segment.confidence && (
          <div className="mt-1 text-xs text-muted-foreground">
            Confidence: {(segment.confidence * 100).toFixed(1)}%
          </div>
        )}
      </CardContent>
      <CardFooter className="px-4 py-2 bg-muted/50">
        {!isEditing ? (
          <Button
            onClick={() => setIsEditing(true)}
            variant="ghost"
            size="sm"
          >
            <Edit className="mr-2 h-4 w-4" />
            Edit
          </Button>
        ) : (
          <>
            <Button
              onClick={handleSave}
              variant="default"
              size="sm"
              className="mr-2"
            >
              <Save className="mr-2 h-4 w-4" />
              Save
            </Button>
            <Button
              onClick={handleCancel}
              variant="ghost"
              size="sm"
            >
              <X className="mr-2 h-4 w-4" />
              Cancel
            </Button>
          </>
        )}
      </CardFooter>
    </Card>
  );
};

export default TranscriptSegment;
