// File: transcription_engine/static/src/components/WaveformPlayer.jsx
import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

const WaveformPlayer = ({ audioUrl, onTimeUpdate, currentTime, duration }) => {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [zoom, setZoom] = useState(50);

  useEffect(() => {
    // Initialize WaveSurfer
    if (waveformRef.current) {
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#4a5568',
        progressColor: '#2b6cb0',
        cursorColor: '#2c5282',
        barWidth: 2,
        barRadius: 3,
        responsive: true,
        height: 80,
        normalize: true,
        backend: 'WebAudio'
      });

      // Load audio file
      wavesurfer.current.load(audioUrl);

      // Setup event handlers
      wavesurfer.current.on('ready', () => {
        console.log('WaveSurfer is ready');
      });

      wavesurfer.current.on('audioprocess', (time) => {
        onTimeUpdate?.(time);
      });

      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
      });

      // Cleanup
      return () => {
        if (wavesurfer.current) {
          wavesurfer.current.destroy();
        }
      };
    }
  }, [audioUrl]);

  // Handle play/pause
  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  // Skip forward/back 5 seconds
  const skip = (direction) => {
    if (wavesurfer.current) {
      const currentTime = wavesurfer.current.getCurrentTime();
      wavesurfer.current.seekTo((currentTime + (direction * 5)) / duration);
    }
  };

  // Update waveform position when currentTime changes externally
  useEffect(() => {
    if (wavesurfer.current && !isPlaying) {
      wavesurfer.current.seekTo(currentTime / duration);
    }
  }, [currentTime, duration]);

  return (
    <div className="w-full space-y-4">
      {/* Waveform container */}
      <div
        ref={waveformRef}
        className="w-full bg-gray-50 rounded-lg p-4 border border-gray-200"
      />

      {/* Controls */}
      <div className="flex items-center justify-center space-x-4">
        <button
          onClick={() => skip(-1)}
          className="p-2 hover:bg-gray-100 rounded-full"
          aria-label="Skip backward"
        >
          <SkipBack className="w-5 h-5" />
        </button>

        <button
          onClick={togglePlay}
          className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <Pause className="w-6 h-6" />
          ) : (
            <Play className="w-6 h-6" />
          )}
        </button>

        <button
          onClick={() => skip(1)}
          className="p-2 hover:bg-gray-100 rounded-full"
          aria-label="Skip forward"
        >
          <SkipForward className="w-5 h-5" />
        </button>

        {/* Zoom control */}
        <input
          type="range"
          min="10"
          max="100"
          value={zoom}
          onChange={(e) => {
            const newZoom = parseInt(e.target.value);
            setZoom(newZoom);
            if (wavesurfer.current) {
              wavesurfer.current.zoom(newZoom);
            }
          }}
          className="w-32"
          aria-label="Zoom level"
        />
      </div>
    </div>
  );
};

export default WaveformPlayer;
