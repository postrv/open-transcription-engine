// File: transcription_engine/static/src/components/AudioUpload.jsx
import React, { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Upload, Loader2, File, CheckCircle2, AlertCircle } from 'lucide-react';
import { Card, CardContent } from './ui/card';

const AudioUpload = ({ onUploadComplete }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const handleFile = async (file) => {
    if (!file) return;

    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/mp3'];
    if (!validTypes.includes(file.type)) {
      setError('Please upload a valid audio file (MP3, WAV, or M4A)');
      setUploadStatus('error');
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB');
      setUploadStatus('error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsUploading(true);
      setError(null);

      const response = await fetch('/api/upload-audio', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload audio file');
      }

      const data = await response.json();
      setUploadStatus('success');
      onUploadComplete?.(data.url);
    } catch (err) {
      setError(err.message || 'Failed to upload audio file');
      setUploadStatus('error');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <Card className={`relative ${dragActive ? 'ring-2 ring-primary' : ''}`}>
      <CardContent className="p-6">
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className="flex flex-col items-center justify-center gap-4"
        >
          <div className={`
            w-full rounded-lg border-2 border-dashed p-6
            transition-all duration-200
            ${dragActive ? 'border-primary bg-primary/5' : 'border-muted'}
            ${uploadStatus === 'success' ? 'border-green-500 bg-green-50' : ''}
            ${uploadStatus === 'error' ? 'border-red-500 bg-red-50' : ''}
          `}>
            <div className="flex flex-col items-center justify-center gap-4">
              {uploadStatus === 'success' ? (
                <CheckCircle2 className="h-10 w-10 text-green-500" />
              ) : uploadStatus === 'error' ? (
                <AlertCircle className="h-10 w-10 text-red-500" />
              ) : (
                <File className="h-10 w-10 text-muted-foreground" />
              )}

              <div className="flex flex-col items-center gap-2">
                <p className="text-sm font-medium">
                  {isUploading ? (
                    'Uploading...'
                  ) : uploadStatus === 'success' ? (
                    'Upload complete!'
                  ) : (
                    <>
                      Drag and drop your audio file here, or{' '}
                      <Button
                        variant="link"
                        className="px-1 text-primary"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        browse
                      </Button>
                    </>
                  )}
                </p>
                <p className="text-xs text-muted-foreground">
                  Supports MP3, WAV, M4A (up to 100MB)
                </p>
              </div>

              {isUploading && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Uploading file...</span>
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="text-sm text-destructive bg-destructive/10 p-3 rounded-md w-full">
              {error}
            </div>
          )}

          <Input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={(e) => handleFile(e.target.files?.[0])}
            className="hidden"
            disabled={isUploading}
          />
        </div>
      </CardContent>
    </Card>
  );
};

export default AudioUpload;
