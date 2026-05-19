import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getSong } from '@/lib/api';
import { Song } from '@/types/song';
import { SongTimeline } from '@/components/SongTimeline';

export function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [song, setSong] = useState<Song | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchSong = useCallback(async () => {
    if (!id) return;
    try {
      const data = await getSong(id);
      setSong(data);
    } catch {
      setError('Could not load song.');
    }
  }, [id]);

  useEffect(() => {
    fetchSong();
  }, [fetchSong]);

  useEffect(() => {
    if (!song || song.status === 'done' || song.status === 'error') return;
    const timer = setInterval(fetchSong, 2000);
    return () => clearInterval(timer);
  }, [song, fetchSong]);

  if (error) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-10">
        <p className="text-destructive">{error}</p>
        <button onClick={() => navigate('/')} className="mt-4 underline text-sm">
          Back
        </button>
      </div>
    );
  }

  if (!song) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-10 text-muted-foreground">
        Loading…
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold truncate">{song.original_name}</h1>
          {song.status === 'done' && song.bpm && (
            <p className="text-muted-foreground text-sm mt-1">
              {song.bpm.toFixed(1)} BPM · {song.music_key ?? '?'} ·{' '}
              {song.duration_sec ? `${Math.round(song.duration_sec)}s` : ''}
            </p>
          )}
        </div>
        <button
          onClick={() => navigate('/')}
          className="text-sm underline text-muted-foreground"
        >
          Back
        </button>
      </div>

      {(song.status === 'queued' || song.status === 'processing') && (
        <div className="rounded-lg border p-8 text-center text-muted-foreground space-y-2">
          <div className="text-2xl animate-spin inline-block">⟳</div>
          <p>{song.status === 'queued' ? 'Waiting in queue…' : 'Analyzing audio…'}</p>
          <p className="text-xs">This takes 1–3 minutes. This page updates automatically.</p>
        </div>
      )}

      {song.status === 'error' && (
        <div className="rounded-lg border border-destructive p-6 text-destructive">
          Analysis failed: {song.error_message ?? 'unknown error'}
        </div>
      )}

      {song.status === 'done' &&
        song.analysis &&
        Object.values(song.analysis.stems).some(
          (s) => !('regions' in (s as object))
        ) && (
          <div className="rounded-lg border p-4 text-sm text-muted-foreground">
            Older analysis format detected. Re-upload this song to view the new region layout.
          </div>
        )}

      {song.status === 'done' && song.analysis && song.analysis.stems && (
        <SongTimeline songId={song.id} analysis={song.analysis} />
      )}
    </div>
  );
}
