import pool from '../db/connect.js';
import { analyzeAudio } from '../services/audioClient.js';
import logger from '../utils/logger.js';

const POLL_INTERVAL_MS = 3000;
const MAX_ATTEMPTS = 3;

async function processNextJob(): Promise<void> {
  const client = await pool.connect();
  let jobId: string | undefined;
  let songId: string | undefined;
  let storageKey: string | undefined;

  try {
    await client.query('BEGIN');
    const { rows } = await client.query<{
      id: string;
      song_id: string;
      storage_key: string;
      attempts: number;
    }>(
      `
      SELECT j.id, j.song_id, j.attempts, s.storage_key
      FROM song_jobs j
      JOIN songs s ON s.id = j.song_id
      WHERE j.status = 'queued' AND j.attempts < $1
      ORDER BY j.created_at
      LIMIT 1
      FOR UPDATE OF j SKIP LOCKED
    `,
      [MAX_ATTEMPTS]
    );

    if (rows.length === 0) {
      await client.query('COMMIT');
      client.release();
      return;
    }

    ({ id: jobId, song_id: songId, storage_key: storageKey } = rows[0]);

    await client.query(
      `UPDATE song_jobs SET status='processing', attempts=attempts+1, started_at=NOW() WHERE id=$1`,
      [jobId]
    );
    await client.query(`UPDATE songs SET status='processing' WHERE id=$1`, [
      songId,
    ]);
    await client.query('COMMIT');
  } catch (err) {
    await client.query('ROLLBACK').catch(() => {});
    client.release();
    logger.error({ err }, 'Job worker DB error during acquire');
    return;
  }

  client.release();

  try {
    logger.info({ jobId, songId }, 'Processing audio job');
    const result = await analyzeAudio(songId!, storageKey!);

    await pool.query(
      `INSERT INTO song_analysis (song_id, result_json) VALUES ($1, $2)
       ON CONFLICT DO NOTHING`,
      [songId, JSON.stringify(result)]
    );
    await pool.query(
      `UPDATE songs SET status='done', bpm=$1, music_key=$2, duration_sec=$3 WHERE id=$4`,
      [
        result.bpm || null,
        result.key !== 'unknown' ? result.key : null,
        result.duration_sec,
        songId,
      ]
    );
    await pool.query(
      `UPDATE song_jobs SET status='done', finished_at=NOW() WHERE id=$1`,
      [jobId]
    );
    logger.info({ jobId, songId }, 'Audio job completed');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    logger.error({ err, jobId, songId }, 'Audio job failed');
    await pool
      .query(
        `UPDATE song_jobs SET status='queued', error=$1, started_at=NULL WHERE id=$2`,
        [message, jobId]
      )
      .catch(() => {});
    await pool
      .query(
        `UPDATE song_jobs SET status='error'
       WHERE id=$1 AND attempts >= $2`,
        [jobId, MAX_ATTEMPTS]
      )
      .catch(() => {});
    await pool
      .query(
        `UPDATE songs SET status='error', error_message=$1
       WHERE id=$2 AND NOT EXISTS (
         SELECT 1 FROM song_jobs WHERE song_id=$2 AND status='queued'
       )`,
        [message, songId]
      )
      .catch(() => {});
  }
}

export function startJobWorker(): void {
  logger.info('Job worker started');
  setInterval(() => {
    processNextJob().catch((err) =>
      logger.error({ err }, 'Unexpected error in job worker')
    );
  }, POLL_INTERVAL_MS);
}
