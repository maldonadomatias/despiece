import { useRef, useState, DragEvent, ChangeEvent } from 'react';
import { uploadSong } from '@/lib/api';

const MAX_MB = 200;
const ALLOWED_EXT = ['.mp3', '.wav', '.flac'];

interface Props {
  onUploaded: (id: string) => void;
}

export function AudioUpload({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function validate(file: File): string | null {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) return 'Only mp3, wav, and flac allowed';
    if (file.size > MAX_MB * 1024 * 1024) return `File must be under ${MAX_MB} MB`;
    return null;
  }

  async function handleFile(file: File) {
    const err = validate(file);
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    setUploading(true);
    try {
      const { id } = await uploadSong(file);
      onUploaded(id);
    } catch {
      setError('Upload failed. Try again.');
    } finally {
      setUploading(false);
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        flex flex-col items-center justify-center gap-3 p-12 rounded-xl border-2 border-dashed cursor-pointer transition-colors
        ${dragging ? 'border-primary bg-primary/5' : 'border-muted-foreground/30 hover:border-primary/50'}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac"
        className="hidden"
        onChange={onChange}
      />
      <div className="text-4xl">🎵</div>
      <p className="text-sm text-muted-foreground text-center">
        {uploading ? 'Uploading…' : 'Drop an audio file here, or click to browse'}
      </p>
      <p className="text-xs text-muted-foreground/60">mp3 · wav · flac · max 200 MB</p>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
