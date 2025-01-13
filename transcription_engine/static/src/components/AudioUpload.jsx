import React, { useState, useRef, useCallback } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Upload, File, XCircle, Loader2 } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import ProcessingStatus from './ProcessingStatus';
import { cn } from '@/lib/utils';

const VALID_TYPES = ['audio/mpeg', 'audio/wav', 'audio/m4a', 'audio/mp3'];
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

const AudioUpload = ({ onUploadComplete }) => {
  const [state, setState] = useState({
    isUploading: false,
    error: null,
    dragActive: false,
    jobId: null,
    uploadProgress: 0,
    file: null
  });

  const fileInputRef = useRef(null);

  const resetState = () => {
    setState(prev => ({
      ...prev,
      isUploading: false,
      error: null,
      uploadProgress: 0,
      file: null
    }));
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const validateFile = useCallback((file) => {
    if (!file) return 'No file selected';
    if (!VALID_TYPES.includes(file.type)) {
      return 'Please upload a valid audio file (MP3, WAV, or M4A)';
    }
    if (file.size > MAX_FILE_SIZE) {
      return 'File size must be less than 100MB';
    }
    return null;
  }, []);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setState(prev => ({ ...prev, dragActive: true }));
    } else if (e.type === "dragleave") {
      setState(prev => ({ ...prev, dragActive: false }));
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();

    setState(prev => ({ ...prev, dragActive: false }));
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, []);

  const handleFile = async (file) => {
    const error = validateFile(file);
    if (error) {
      setState(prev => ({ ...prev, error }));
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setState(prev => ({
      ...prev,
      isUploading: true,
      error: null,
      jobId: null,
      file
    }));

    try {
      const response = await fetch('/api/upload-audio', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload audio file');
      }

      const data = await response.json();
      setState(prev => ({
        ...prev,
        jobId: data.job_id,
        isUploading: false
      }));

      onUploadComplete?.(data.url, data.job_id);
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: err.message || 'Failed to upload audio file',
        isUploading: false
      }));
    }
  };

  // Remove the logger import entirely and modify the handler:
const handleProcessingComplete = useCallback(async (outputPath) => {
    if (!outputPath) {
      console.error('No output path received');
      return;
    }

    try {
      // Extract job ID from output path
      const jobId = outputPath.split('/').pop().split('.')[0];

      // Add delay to ensure file is written
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Fetch the transcript for this specific job
      const response = await fetch(`/api/transcript/${jobId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch transcript: ${response.statusText}`);
      }

      const data = await response.json();
      console.log(`Loaded transcript for job ${jobId} with ${data.length} segments`);

      // Trigger parent component update with memoized value
      onUploadComplete?.(
        state.file ? URL.createObjectURL(state.file) : null,
        jobId,
        data
      );

    } catch (error) {
      console.error('Error loading transcript:', error);
      setState(prev => ({
        ...prev,
        error: `Failed to load transcript: ${error.message}`
      }));
    }
}, [state.file, onUploadComplete]);

  const { isUploading, error, dragActive, jobId, file } = state;

  return (
    <div className="space-y-4">
      <Card
        className={cn(
          "relative transition-all duration-200",
          dragActive && "ring-2 ring-primary",
          error && "ring-2 ring-destructive"
        )}
      >
        <CardContent className="p-6">
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className="flex flex-col items-center justify-center gap-4"
          >
            <div className={cn(
              "w-full rounded-lg border-2 border-dashed p-6",
              "transition-all duration-200",
              dragActive ? "border-primary bg-primary/5" : "border-muted",
              error ? "border-destructive bg-destructive/5" : "",
              isUploading && "opacity-50"
            )}>
              <div className="flex flex-col items-center justify-center gap-4">
                <div className={cn(
                  "flex items-center justify-center size-12 rounded-full",
                  error ? "bg-destructive/10" : "bg-primary/10"
                )}>
                  {isUploading ? (
                    <Loader2 className="h-6 w-6 animate-spin text-primary" />
                  ) : error ? (
                    <XCircle className="h-6 w-6 text-destructive" />
                  ) : (
                    <File className="h-6 w-6 text-primary" />
                  )}
                </div>

                <div className="flex flex-col items-center gap-2 text-center">
                  {file && !error ? (
                    <p className="text-sm font-medium">
                      {file.name}
                    </p>
                  ) : (
                    <p className="text-sm font-medium">
                      {isUploading ? (
                        'Uploading...'
                      ) : (
                        <>
                          Drag and drop your audio file here, or{' '}
                          <Button
                            variant="link"
                            className="px-1 text-primary"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploading}
                          >
                            browse
                          </Button>
                        </>
                      )}
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Supports MP3, WAV, M4A (up to 100MB)
                  </p>
                </div>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 p-3 rounded-md w-full">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                <span>{error}</span>
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

      {jobId && (
        <ProcessingStatus
          jobId={jobId}
          onComplete={handleProcessingComplete}
        />
      )}
    </div>
  );
};

export default AudioUpload;
