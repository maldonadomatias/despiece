import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AudioUpload } from '@/components/AudioUpload';
import { SongCard } from '@/components/SongCard';
import { listSongs } from '@/lib/api';
import { Song } from '@/types/song';

export function HomePage() {
  const navigate = useNavigate();
  const [songs, setSongs] = useState<Song[]>([]);

  async function refresh() {
    try {
      setSongs(await listSongs());
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  function onUploaded(id: string) {
    navigate(`/songs/${id}`);
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-10 space-y-10">
      <div>
        <h1 className="text-3xl font-bold mb-2">Song Analyzer</h1>
        <p className="text-muted-foreground text-sm">
          Upload a track to visualize its structure — stems, sections, BPM and key.
        </p>
      </div>
      <AudioUpload onUploaded={onUploaded} />
      {songs.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-4">Previous analyses</h2>
          <div className="space-y-2">
            {songs.map((s) => (
              <SongCard
                key={s.id}
                song={s}
                onClick={() => navigate(`/songs/${s.id}`)}
                onDelete={refresh}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
