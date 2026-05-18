export type SongStatus = 'queued' | 'processing' | 'done' | 'error';

export interface Song {
  id: string;
  original_name: string;
  duration_sec: number | null;
  bpm: number | null;
  music_key: string | null;
  status: SongStatus;
  error_message: string | null;
  created_at: string;
  analysis?: AnalysisResult;
}

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  sections: Section[];
  stems: Record<string, StemData>;
}

export interface Section {
  label: string;
  start_sec: number;
  end_sec: number;
}

export interface StemData {
  envelope: [number, number][]; // [time_sec, energy_0_1]
}
