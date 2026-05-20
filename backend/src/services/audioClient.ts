import { Agent, fetch } from 'undici';
import { AnalysisResult } from '../domain/song.js';

const AUDIO_SERVICE_URL =
  process.env.AUDIO_SERVICE_URL ?? 'http://localhost:8000';

// Full pipeline (Phase A ensemble + B2 drum hits + B1 chord track) takes
// 30-45 min on CPU per the design budget. undici's default headersTimeout
// is 5 min; we set 90 min to give the pipeline headroom plus retries.
const ANALYZE_TIMEOUT_MS = 90 * 60 * 1000;

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
