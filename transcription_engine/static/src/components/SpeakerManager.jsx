// File: transcription_engine/static/src/components/SpeakerManager.jsx
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { User, Users } from 'lucide-react';

const SpeakerManager = ({ segments, onSpeakerUpdate }) => {
  const [speakers, setSpeakers] = useState(new Set());
  const [selectedSpeaker, setSelectedSpeaker] = useState(null);
  const speakerColors = {
    'SPEAKER_1': 'bg-blue-100 border-blue-300',
    'SPEAKER_2': 'bg-green-100 border-green-300',
    'SPEAKER_3': 'bg-yellow-100 border-yellow-300',
    'SPEAKER_4': 'bg-purple-100 border-purple-300',
    'SPEAKER_5': 'bg-pink-100 border-pink-300',
  };

  useEffect(() => {
    // Extract unique speakers from segments
    const uniqueSpeakers = new Set(
      segments
        .map(seg => seg.speaker_id)
        .filter(Boolean)
    );
    setSpeakers(uniqueSpeakers);
  }, [segments]);

  const handleSpeakerSelect = (speakerId) => {
    setSelectedSpeaker(speakerId === selectedSpeaker ? null : speakerId);
  };

  const getSpeakerColor = (speakerId) => {
    return speakerColors[speakerId] || 'bg-gray-100 border-gray-300';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5" />
          Speakers
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {speakers.size === 0 ? (
            <div className="text-sm text-muted-foreground">
              No speakers identified yet
            </div>
          ) : (
            Array.from(speakers).map((speakerId) => (
              <Button
                key={speakerId}
                variant="outline"
                className={`w-full justify-start gap-2 ${
                  getSpeakerColor(speakerId)
                } ${
                  selectedSpeaker === speakerId ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() => handleSpeakerSelect(speakerId)}
              >
                <User className="h-4 w-4" />
                {speakerId}
              </Button>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default SpeakerManager;
