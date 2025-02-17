// File: transcription_engine/static/src/components/TranscriptSegment.jsx
import React, { useState } from "react";
import { Card, CardContent, CardFooter } from "./ui/card";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import {
  Edit,
  Save,
  X,
  Clock,
  AlertTriangle,
  Users,
  Volume2
} from 'lucide-react';
import SpeakerSelect from "./ui/speaker-select";

const TranscriptSegment = ({
  segment,
  onSegmentUpdate,
  index,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [isSpeakerEditing, setIsSpeakerEditing] = useState(false);
  const [draftText, setDraftText] = useState(segment.text);

  const handleSave = () => {
    setIsEditing(false);
    onSegmentUpdate?.(index, {
      ...segment,
      text: draftText,
    });
  };

  const handleCancel = () => {
    setIsEditing(false);
    setDraftText(segment.text);
  };

  const handleSpeakerUpdate = (newSpeakerId) => {
    onSegmentUpdate?.(index, {
      ...segment,
      speaker_id: newSpeakerId,
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'bg-green-500';
    if (confidence >= 0.7) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const renderDiarizationBadges = () => {
    const { diarization_data = {} } = segment;
    return (
      <div className="flex gap-2 items-center text-xs">
        {diarization_data.overlap_detected && (
          <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-yellow-100 text-yellow-800">
            <Users className="h-3 w-3" />
            <span>Overlap</span>
          </div>
        )}
        {diarization_data.energy_score > 0 && (
          <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-blue-100 text-blue-800">
            <Volume2 className="h-3 w-3" />
            <span>{Math.round(diarization_data.energy_score * 100)}% Energy</span>
          </div>
        )}
      </div>
    );
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

              <SpeakerSelect
                value={segment.speaker_id}
                onValueChange={handleSpeakerUpdate}
                isEditing={isSpeakerEditing}
                onEditStart={() => setIsSpeakerEditing(true)}
                onEditCancel={() => setIsSpeakerEditing(false)}
                onEditComplete={() => setIsSpeakerEditing(false)}
              />
            </div>
          </div>

          {renderDiarizationBadges()}

          {isEditing ? (
            <Textarea
              value={draftText}
              onChange={(e) => setDraftText(e.target.value)}
              className="mt-2 min-h-[100px] text-base"
              rows={4}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                  handleSave();
                }
                if (e.key === 'Escape') {
                  handleCancel();
                }
              }}
              placeholder="Enter transcript text..."
              autoFocus
            />
          ) : (
            <div
              className="mt-2 text-base leading-relaxed cursor-pointer hover:bg-muted/50 p-2 rounded-md"
              onClick={() => setIsEditing(true)}
            >
              {segment.text}
            </div>
          )}

          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-24">Transcription:</span>
              <div className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getConfidenceColor(segment.confidence)}`}
                  style={{ width: `${segment.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground w-12 text-right">
                {(segment.confidence * 100).toFixed(1)}%
              </span>
            </div>

            {segment.diarization_data?.energy_score > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-24">Voice Energy:</span>
                <div className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-500"
                    style={{ width: `${segment.diarization_data.energy_score * 100}%` }}
                  />
                </div>
                <span className="text-xs text-muted-foreground w-12 text-right">
                  {(segment.diarization_data.energy_score * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
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
              Save (Ctrl+Enter)
            </Button>
            <Button
              onClick={handleCancel}
              variant="ghost"
              size="sm"
              className="gap-2 text-destructive hover:bg-destructive/10"
            >
              <X className="h-4 w-4" />
              Cancel (Esc)
            </Button>
          </div>
        )}
      </CardFooter>
    </Card>
  );
};

export default TranscriptSegment;