// File: transcription_engine/static/src/components/ui/speaker-select.jsx
import React from 'react';
import { Button } from './button';
import { User, Users, Edit2, Check, X } from 'lucide-react';
import { Input } from './input';

const SpeakerSelect = ({
  value,
  onValueChange,
  isEditing,
  onEditStart,
  onEditCancel,
  onEditComplete,
  className
}) => {
  const [draftValue, setDraftValue] = React.useState(value || '');

  const handleValueChange = (e) => {
    setDraftValue(e.target.value);
  };

  const handleEditComplete = () => {
    onValueChange(draftValue);
    onEditComplete();
  };

  const handleEditCancel = () => {
    setDraftValue(value || '');
    onEditCancel();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleEditComplete();
    } else if (e.key === 'Escape') {
      handleEditCancel();
    }
  };

  React.useEffect(() => {
    setDraftValue(value || '');
  }, [value]);

  if (isEditing) {
    return (
      <div className="flex items-center gap-2">
        <User className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        <Input
          type="text"
          value={draftValue}
          onChange={handleValueChange}
          onKeyDown={handleKeyDown}
          className="h-8 w-40"
          placeholder="Speaker ID"
          autoFocus
        />
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-green-600 hover:text-green-700 hover:bg-green-100"
            onClick={handleEditComplete}
          >
            <Check className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-red-600 hover:text-red-700 hover:bg-red-100"
            onClick={handleEditCancel}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={onEditStart}
      className="gap-2 h-8 hover:bg-muted"
    >
      <User className="h-4 w-4" />
      <span>{value || 'Unknown Speaker'}</span>
      <Edit2 className="h-3 w-3 text-muted-foreground" />
    </Button>
  );
};

export default SpeakerSelect;