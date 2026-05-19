import pool from '../db/connect.js';

async function main() {
  console.log(
    'Resetting song_analysis + setting songs.status=queued for all rows...',
  );
  await pool.query('BEGIN');
  try {
    await pool.query('DELETE FROM song_analysis');
    const res = await pool.query(
      "UPDATE songs SET status = 'queued', error_message = NULL",
    );
    await pool.query('COMMIT');
    console.log(`Done. ${res.rowCount ?? 0} songs requeued.`);
  } catch (err) {
    await pool.query('ROLLBACK');
    console.error('Failed:', err);
    process.exitCode = 1;
  } finally {
    await pool.end();
  }
}

main();
