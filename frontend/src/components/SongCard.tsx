import { Song } from '@/types/song';
import { deleteSong } from '@/lib/api';

const STATUS_LABEL: Record<string, string> = {
  queued: 'Queued…',
  processing: 'Analyzing…',
  done: 'Done',
  error: 'Error',
};

interface Props {
  song: Song;
  onClick: () => void;
  onDelete?: () => void;
}

export function SongCard({ song, onClick, onDelete }: Props) {
  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm(`Delete "${song.original_name}"?`)) return;
    await deleteSong(song.id).catch(() => {});
    onDelete?.();
  }

  return (
    <div
      onClick={onClick}
      className="flex items-center justify-between px-4 py-3 rounded-lg border hover:bg-muted/50 cursor-pointer transition-colors"
    >
      <div className="min-w-0">
        <p className="font-medium truncate">{song.original_name}</p>
        <p className="text-xs text-muted-foreground">
          {song.bpm ? `${song.bpm.toFixed(1)} BPM` : '—'}
          {song.music_key ? ` · ${song.music_key}` : ''}
          {song.duration_sec ? ` · ${Math.round(song.duration_sec)}s` : ''}
        </p>
      </div>
      <div className="flex items-center gap-2">
        <span
          className={`text-xs font-medium px-2 py-1 rounded-full ${
            song.status === 'done'
              ? 'bg-green-100 text-green-700'
              : song.status === 'error'
                ? 'bg-red-100 text-red-700'
                : 'bg-yellow-100 text-yellow-700'
          }`}
        >
          {STATUS_LABEL[song.status] ?? song.status}
        </span>
        {onDelete && (
          <button
            onClick={handleDelete}
            className="text-muted-foreground hover:text-destructive transition-colors p-1"
            title="Delete"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
