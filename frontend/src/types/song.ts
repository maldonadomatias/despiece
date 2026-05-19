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

export type SubLabel = 'lead' | 'pad' | 'synth' | 'strings' | 'fx' | 'other_misc';

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  bar_grid: number[];
  sections: Section[];
  stems: Record<string, StemAnalysis>;
  analysis_version: number;
}

export interface Section {
  label: string;
  start_sec: number;
  end_sec: number;
}

export interface StemAnalysis {
  audio_key: string | null;
  regions: StemRegion[];
}

export interface StemRegion {
  start_sec: number;
  end_sec: number;
  envelope: [number, number][]; // [t_relative_sec, energy_0_1]
  sub_label?: SubLabel;
  sub_label_confidence?: number;
}
