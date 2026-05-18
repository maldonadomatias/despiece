import { AnalysisResult } from '../domain/song.js';

const AUDIO_SERVICE_URL =
  process.env.AUDIO_SERVICE_URL ?? 'http://localhost:8000';

// Demucs on CPU can take 5-15 min for a typical song
const ANALYZE_TIMEOUT_MS = 30 * 60 * 1000;

export async function analyzeAudio(storageKey: string): Promise<AnalysisResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS);

  try {
    const res = await fetch(`${AUDIO_SERVICE_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ storage_key: storageKey }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => 'unknown error');
      throw new Error(`Audio service error ${res.status}: ${text}`);
    }

    return (await res.json()) as AnalysisResult;
  } finally {
    clearTimeout(timer);
  }
}
