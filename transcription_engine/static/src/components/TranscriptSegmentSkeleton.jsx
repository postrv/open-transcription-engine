// File: transcription_engine/static/src/components/TranscriptSegmentSkeleton.jsx

import React from 'react';
import { Card, CardContent, CardFooter } from './ui/card';

const TranscriptSegmentSkeleton = () => {
  return (
    <Card className="w-full animate-pulse">
      <CardContent className="p-4">
        <div className="flex justify-between items-start mb-3">
          <div className="h-4 w-24 bg-muted rounded" />
          <div className="h-6 w-20 bg-muted rounded" />
        </div>
        <div className="space-y-2">
          <div className="h-4 w-3/4 bg-muted rounded" />
          <div className="h-4 w-full bg-muted rounded" />
          <div className="h-4 w-2/3 bg-muted rounded" />
        </div>
        <div className="mt-4 flex items-center gap-2">
          <div className="h-2 w-16 bg-muted rounded" />
          <div className="h-2 w-12 bg-muted rounded" />
        </div>
      </CardContent>
      <CardFooter className="px-4 py-2 border-t">
        <div className="h-8 w-20 bg-muted rounded" />
      </CardFooter>
    </Card>
  );
};

export default TranscriptSegmentSkeleton;
