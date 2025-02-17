import React, { useState, useEffect } from "react";
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
  Volume2,
  Wand2,
  Loader2,
  CheckCircle2,
  Sparkles,
} from "lucide-react";
import SpeakerSelect from "./ui/speaker-select";
import { cn } from "@/lib/utils";

const TranscriptSegment = ({
  segment,
  onSegmentUpdate,
  index,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [isSpeakerEditing, setIsSpeakerEditing] = useState(false);
  const [isCorrectingWithAI, setIsCorrectingWithAI] = useState(false);
  const [draftText, setDraftText] = useState(segment.text);
  const [showCelebration, setShowCelebration] = useState(false);

  // Track confidence changes for animation
  const [previousConfidence, setPreviousConfidence] = useState(segment.confidence);
  const [confidenceImproved, setConfidenceImproved] = useState(false);

  useEffect(() => {
    if (segment.confidence > previousConfidence) {
      setConfidenceImproved(true);
      if (segment.confidence >= 0.85) {
        setShowCelebration(true);
        setTimeout(() => setShowCelebration(false), 2000); // Hide celebration after 2s
      }
    }
    setPreviousConfidence(segment.confidence);
  }, [segment.confidence, previousConfidence]);

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

  const handleAICorrection = async () => {
    try {
      setIsCorrectingWithAI(true);
      const response = await fetch("/api/correct-segment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          segment_id: index,
          text: segment.text,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to correct segment");
      }

      const data = await response.json();

      // Only update if there were actual changes
      if (data.corrected_text !== segment.text) {
        onSegmentUpdate?.(index, {
          ...segment,
          text: data.corrected_text,
          ai_correction_confidence: data.confidence,
          last_correction: new Date().toISOString(),
        });
      }
    } catch (error) {
      console.error("Error correcting segment:", error);
    } finally {
      setIsCorrectingWithAI(false);
    }
  };

  const handleSpeakerUpdate = (newSpeakerId) => {
    onSegmentUpdate?.(index, {
      ...segment,
      speaker_id: newSpeakerId,
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return "bg-green-500";
    if (confidence >= 0.7) return "bg-yellow-500";
    return "bg-red-500";
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
    <Card
      className={cn(
        "w-full transition-all duration-300",
        "hover:shadow-md",
        isEditing && "ring-2 ring-primary",
        confidenceImproved && "animate-pulse",
        showCelebration && "ring-2 ring-green-500"
      )}
    >
      <CardContent className="p-4">
        <div className="flex flex-col gap-3">
          <div className="flex justify-between items-start">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>{`${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s`}</span>
            </div>

            <div className="flex items-center gap-2">
              {/* Confidence Indicator */}
              <div className="flex items-center gap-1">
                {segment.confidence || segment.ai_correction_confidence >= 0.85 ? (
                  <div className="flex items-center gap-1 text-green-500">
                    <CheckCircle2 className="h-4 w-4" />
                    <span className="text-xs font-medium">High confidence</span>
                    {showCelebration && <Sparkles className="h-4 w-4 animate-bounce" />}
                  </div>
                ) : segment.confidence < 0.7 ? (
                  <div className="flex items-center gap-1 text-yellow-500">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="text-xs">Low confidence</span>
                  </div>
                ) : null}
              </div>

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
                if (e.key === "Enter" && e.ctrlKey) {
                  handleSave();
                }
                if (e.key === "Escape") {
                  handleCancel();
                }
              }}
              placeholder="Enter transcript text..."
              autoFocus
            />
          ) : (
            <div
              className={cn(
                "mt-2 text-base leading-relaxed cursor-pointer p-2 rounded-md",
                "hover:bg-muted/50",
                segment.confidence >= 0.85 && "bg-green-50 dark:bg-green-950/10"
              )}
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
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    getConfidenceColor(segment.confidence)
                  )}
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

            {segment.ai_correction_confidence > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-24">AI Correction:</span>
                <div className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      segment.ai_correction_confidence >= 0.85 ? "bg-green-500" : "bg-purple-500"
                    )}
                    style={{ width: `${segment.ai_correction_confidence * 100}%` }}
                  />
                </div>
                <span className="text-xs text-muted-foreground w-12 text-right">
                  {(segment.ai_correction_confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        </div>
      </CardContent>

      <CardFooter className="px-4 py-2 border-t">
        <div className="flex gap-2">
          {!isEditing ? (
            <>
              <Button
                onClick={() => setIsEditing(true)}
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-foreground gap-2"
              >
                <Edit className="h-4 w-4" />
                Edit
              </Button>
              <Button
                onClick={handleAICorrection}
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-foreground gap-2"
                disabled={isCorrectingWithAI}
              >
                {isCorrectingWithAI ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Wand2 className="h-4 w-4" />
                )}
                Fix with AI
              </Button>
            </>
          ) : (
            <>
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
            </>
          )}
        </div>
      </CardFooter>
    </Card>
  );
};

export default TranscriptSegment;