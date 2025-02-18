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
  Scissors,
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
  const [selection, setSelection] = useState(null);
  const [showSplitUI, setShowSplitUI] = useState(false);

  // Track confidence changes for animation
  const [previousConfidence, setPreviousConfidence] = useState(segment.confidence);
  const [confidenceImproved, setConfidenceImproved] = useState(false);

  const getEffectiveConfidence = () => {
    // Use the highest confidence value between transcription and AI correction
    const transcriptionConfidence = segment.confidence || 0;
    const aiCorrection = segment.ai_correction_confidence || 0;
    return Math.max(transcriptionConfidence, aiCorrection);
  };

  const isHighConfidence = getEffectiveConfidence() >= 0.85;

  useEffect(() => {
    if (segment.confidence > previousConfidence) {
      setConfidenceImproved(true);
      if (segment.confidence >= 0.85) {
        setShowCelebration(true);
        setTimeout(() => setShowCelebration(false), 2000);
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
    setSelection(null);
    setShowSplitUI(false);
  };

  const handleTextSelection = () => {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (selectedText.length > 0) {
      // Get the text node containing the selection
      let container = selection.anchorNode;

      // If we selected the element instead of text node, adjust accordingly
      if (container.nodeType !== Node.TEXT_NODE) {
        container = container.firstChild;
      }

      // Ensure we have valid text content
      if (!container || !container.textContent) {
        console.error('Invalid selection container:', container);
        return;
      }

      const start = Math.min(selection.anchorOffset, selection.focusOffset);
      const end = Math.max(selection.anchorOffset, selection.focusOffset);

      console.log('Selection details:', {
        selectedText,
        start,
        end,
        container: container.textContent
      });

      setSelection({
        text: selectedText,
        start,
        end
      });
      setShowSplitUI(true);
    } else {
      setSelection(null);
      setShowSplitUI(false);
    }
  };

  const handleSplit = () => {
    if (!selection) return;

    // Ensure we have valid start/end times
    const startTime = parseFloat(segment.start);
    const endTime = parseFloat(segment.end);

    if (isNaN(startTime) || isNaN(endTime)) {
      console.error('Invalid segment timing:', { start: segment.start, end: segment.end });
      return;
    }

    const totalDuration = endTime - startTime;
    const text = segment.text;
    const totalChars = text.length;

    // Calculate time positions based on character positions
    const selectionStartPct = selection.start / totalChars;
    const selectionEndPct = selection.end / totalChars;

    // Calculate absolute timestamps
    const selectionStartTime = startTime + (totalDuration * selectionStartPct);
    const selectionEndTime = startTime + (totalDuration * selectionEndPct);

    // Create segments array
    const newSegments = [];

    // Pre-selection segment (if there's text before selection)
    if (selection.start > 0) {
      newSegments.push({
        ...segment,
        text: text.slice(0, selection.start).trim(),
        start: startTime,
        end: selectionStartTime,
      });
    }

    // Selected segment
    newSegments.push({
      ...segment,
      text: selection.text.trim(),
      start: selectionStartTime,
      end: selectionEndTime,
    });

    // Post-selection segment (if there's text after selection)
    if (selection.end < text.length) {
      newSegments.push({
        ...segment,
        text: text.slice(selection.end).trim(),
        start: selectionEndTime,
        end: endTime,
      });
    }

    // Validate all segments have proper timestamps and non-empty text
    if (newSegments.some(seg =>
      isNaN(seg.start) ||
      isNaN(seg.end) ||
      seg.start >= seg.end ||
      !seg.text.trim()
    )) {
      console.error('Invalid segment generated:', newSegments);
      return;
    }

    // Log the split operation for debugging
    console.log('Split operation:', {
      original: segment,
      selection,
      newSegments
    });

    // Notify parent of the split
    onSegmentUpdate?.(index, newSegments);

    // Reset selection state
    setSelection(null);
    setShowSplitUI(false);
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

  const renderSplitPreview = () => {
    if (!selection || !segment) return null;

    const text = segment.text;
    const beforeText = text.slice(0, selection.start).trim();
    const selectedText = selection.text.trim();
    const afterText = text.slice(selection.end).trim();

    return (
      <div className="flex flex-col gap-2 p-2 text-sm text-muted-foreground">
        {beforeText && (
          <div className="flex gap-2">
            <span className="font-medium">Before:</span>
            <span className="italic">{beforeText}</span>
          </div>
        )}
        <div className="flex gap-2">
          <span className="font-medium">Selection:</span>
          <span className="italic text-primary">{selectedText}</span>
        </div>
        {afterText && (
          <div className="flex gap-2">
            <span className="font-medium">After:</span>
            <span className="italic">{afterText}</span>
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
                {isHighConfidence ? (
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
                // Support both Ctrl and Cmd (Meta) key for cross-platform
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  handleSave();
                }
                if (e.key === "Escape") {
                  e.preventDefault();
                  handleCancel();
                }
              }}
              placeholder="Enter transcript text..."
              autoFocus
            />
          ) : (
            <>
              <div
                className={cn(
                  "mt-2 text-base leading-relaxed cursor-pointer p-2 rounded-md",
                  "hover:bg-muted/50",
                  segment.confidence >= 0.85 && "bg-green-50 dark:bg-green-950/10"
                )}
                onClick={() => setIsEditing(true)}
                onMouseUp={handleTextSelection}
              >
                {segment.text}
              </div>
              {showSplitUI && selection && (
                  <div className="flex flex-col gap-2 mt-2 p-3 bg-muted rounded-md border border-muted-foreground/20">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="text-sm font-medium">Split segment at selection</p>
                        <div className="text-sm text-muted-foreground mt-1">
                          Will create {selection.start > 0 && selection.end < segment.text.length ? "3" : "2"} segments
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setSelection(null);
                          setShowSplitUI(false);
                        }}
                        className="text-muted-foreground hover:text-foreground"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>

                    {renderSplitPreview()}

                    <div className="flex gap-2 text-xs text-muted-foreground mt-1">
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        <span>Total duration: {((segment.end - segment.start) * 100) / 100}s</span>
                      </div>
                    </div>

                    <div className="flex justify-end gap-2 mt-2">
                      <Button
                        onClick={handleSplit}
                        variant="secondary"
                        size="sm"
                        className="gap-2"
                      >
                        <Scissors className="h-4 w-4" />
                        Split segment
                      </Button>
                      <Button
                        onClick={() => {
                          setSelection(null);
                          setShowSplitUI(false);
                        }}
                        variant="ghost"
                        size="sm"
                        className="text-destructive hover:bg-destructive/10"
                      >
                        <X className="h-4 w-4" />
                        Cancel
                      </Button>
                    </div>
                  </div>
              )}
            </>
          )}
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-24">Transcription:</span>
              <div className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    getEffectiveConfidence() >= 0.85 ? "bg-green-500" :
                    getEffectiveConfidence() >= 0.7 ? "bg-yellow-500" :
                    "bg-red-500"
                  )}
                  style={{ width: `${getEffectiveConfidence() * 100}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground w-12 text-right">
                {(getEffectiveConfidence() * 100).toFixed(1)}%
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