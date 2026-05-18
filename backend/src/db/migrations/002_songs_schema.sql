-- Songs: one row per uploaded audio file
CREATE TABLE IF NOT EXISTS songs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  storage_key TEXT NOT NULL,
  original_name TEXT NOT NULL,
  duration_sec NUMERIC(10,3),
  bpm NUMERIC(6,2),
  music_key TEXT,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued','processing','done','error')),
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Job queue: one row per analysis job
CREATE TABLE IF NOT EXISTS song_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued','processing','done','error')),
  attempts INT NOT NULL DEFAULT 0,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_song_jobs_status ON song_jobs(status, created_at)
  WHERE status = 'queued';

-- Analysis results: JSONB payload returned by audio-service
CREATE TABLE IF NOT EXISTS song_analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
  result_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_song_analysis_song_id ON song_analysis(song_id);

CREATE INDEX IF NOT EXISTS idx_songs_status ON songs(status);
CREATE INDEX IF NOT EXISTS idx_song_jobs_song_id ON song_jobs(song_id);
