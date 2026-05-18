import pool from '../db/connect.js';
import { Song, SongAnalysis } from '../domain/song.js';

export async function createSong(
  storageKey: string,
  originalName: string
): Promise<Song> {
  const { rows } = await pool.query<Song>(
    `INSERT INTO songs (storage_key, original_name)
     VALUES ($1, $2) RETURNING *`,
    [storageKey, originalName]
  );
  return rows[0];
}

export async function getSong(id: string): Promise<Song | null> {
  const { rows } = await pool.query<Song>(
    'SELECT * FROM songs WHERE id = $1',
    [id]
  );
  return rows[0] ?? null;
}

export async function listSongs(): Promise<Song[]> {
  const { rows } = await pool.query<Song>(
    'SELECT * FROM songs ORDER BY created_at DESC'
  );
  return rows;
}

export async function getSongAnalysis(
  songId: string
): Promise<SongAnalysis | null> {
  const { rows } = await pool.query<SongAnalysis>(
    'SELECT * FROM song_analysis WHERE song_id = $1',
    [songId]
  );
  return rows[0] ?? null;
}

export async function createJob(songId: string): Promise<string> {
  const { rows } = await pool.query<{ id: string }>(
    `INSERT INTO song_jobs (song_id) VALUES ($1) RETURNING id`,
    [songId]
  );
  return rows[0].id;
}

export async function deleteSong(id: string): Promise<string | null> {
  const { rows } = await pool.query<{ storage_key: string }>(
    'DELETE FROM songs WHERE id = $1 RETURNING storage_key',
    [id]
  );
  return rows[0]?.storage_key ?? null;
}
