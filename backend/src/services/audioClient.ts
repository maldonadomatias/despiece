import { AnalysisResult } from '../domain/song.js';
import logger from '../utils/logger.js';

const AUDIO_SERVICE_URL =
  process.env.AUDIO_SERVICE_URL ?? 'http://localhost:8000';

export async function analyzeAudio(storageKey: string): Promise<AnalysisResult> {
  const res = await fetch(`${AUDIO_SERVICE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storage_key: storageKey }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'unknown error');
    throw new Error(`Audio service error ${res.status}: ${text}`);
  }

  return res.json() as Promise<AnalysisResult>;
}
