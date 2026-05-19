import { Agent, fetch } from 'undici';
import { AnalysisResult } from '../domain/song.js';

const AUDIO_SERVICE_URL =
  process.env.AUDIO_SERVICE_URL ?? 'http://localhost:8000';

// Demucs on CPU can take 5-15 min for a typical song.
// undici's default headersTimeout is 5 min, which trips long Demucs jobs.
const ANALYZE_TIMEOUT_MS = 30 * 60 * 1000;

const longTimeoutAgent = new Agent({
  headersTimeout: ANALYZE_TIMEOUT_MS,
  bodyTimeout: ANALYZE_TIMEOUT_MS,
  keepAliveTimeout: ANALYZE_TIMEOUT_MS,
});

export async function analyzeAudio(songId: string, storageKey: string): Promise<AnalysisResult> {
  const res = await fetch(`${AUDIO_SERVICE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storage_key: storageKey, song_id: songId }),
    dispatcher: longTimeoutAgent,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'unknown error');
    throw new Error(`Audio service error ${res.status}: ${text}`);
  }

  return (await res.json()) as AnalysisResult;
}
