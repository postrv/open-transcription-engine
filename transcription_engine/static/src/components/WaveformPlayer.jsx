import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause, SkipBack, SkipForward, ZoomIn, ZoomOut } from 'lucide-react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';

const WaveformPlayer = ({ audioUrl, onTimeUpdate, currentTime, duration }) => {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [zoom, setZoom] = useState(50);

  useEffect(() => {
    if (waveformRef.current) {
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'hsl(var(--muted-foreground))',
        progressColor: 'hsl(var(--primary))',
        cursorColor: 'hsl(var(--primary))',
        barWidth: 2,
        barRadius: 3,
        responsive: true,
        height: 80,
        normalize: true,
        backend: 'WebAudio',
        barGap: 2,
        minPxPerSec: 50
      });

      wavesurfer.current.load(audioUrl);

      wavesurfer.current.on('ready', () => {
        console.log('WaveSurfer is ready');
      });

      wavesurfer.current.on('audioprocess', (time) => {
        onTimeUpdate?.(time);
      });

      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
      });

      return () => {
        if (wavesurfer.current) {
          wavesurfer.current.destroy();
        }
      };
    }
  }, [audioUrl]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  const skip = (direction) => {
    if (wavesurfer.current) {
      const currentTime = wavesurfer.current.getCurrentTime();
      wavesurfer.current.seekTo((currentTime + (direction * 5)) / duration);
    }
  };

  useEffect(() => {
    if (wavesurfer.current && !isPlaying) {
      wavesurfer.current.seekTo(currentTime / duration);
    }
  }, [currentTime, duration]);

  const handleZoom = (direction) => {
    const newZoom = Math.max(10, Math.min(100, zoom + direction * 10));
    setZoom(newZoom);
    if (wavesurfer.current) {
      wavesurfer.current.zoom(newZoom);
    }
  };

  return (
    <div className="space-y-4">
      <div
        ref={waveformRef}
        className="w-full bg-muted rounded-lg p-4"
      />

      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Button onClick={() => skip(-1)} size="icon" variant="ghost">
            <SkipBack className="h-4 w-4" />
          </Button>
          <Button onClick={togglePlay} size="icon">
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button onClick={() => skip(1)} size="icon" variant="ghost">
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center space-x-2">
          <Button onClick={() => handleZoom(-1)} size="icon" variant="ghost">
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Slider
            value={[zoom]}
            onValueChange={([value]) => handleZoom(value - zoom)}
            min={10}
            max={100}
            step={1}
            className="w-32"
          />
          <Button onClick={() => handleZoom(1)} size="icon" variant="ghost">
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default WaveformPlayer;
