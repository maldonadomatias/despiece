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

export type AnalysisVersion = 1 | 2 | 3 | 4;

export type ChordLabel =
  | 'C' | 'C#' | 'D' | 'D#' | 'E' | 'F' | 'F#' | 'G' | 'G#' | 'A' | 'A#' | 'B'
  | 'Cm' | 'C#m' | 'Dm' | 'D#m' | 'Em' | 'Fm' | 'F#m' | 'Gm' | 'G#m' | 'Am' | 'A#m' | 'Bm'
  | 'N';

export interface Chord {
  start_sec: number;
  end_sec: number;
  label: ChordLabel;
}

export type DrumHitClass = 'kick' | 'snare' | 'hihat' | 'cymbal' | 'unknown';

export interface DrumHit {
  t_sec: number;
  velocity: number;    // 0-1
  confidence: number;  // 0-1
}

export interface DrumHits {
  kick: DrumHit[];
  snare: DrumHit[];
  hihat: DrumHit[];
  cymbal: DrumHit[];
  unknown: DrumHit[];
}

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  bar_grid: number[];
  sections: Section[];
  stems: Record<string, StemAnalysis>;
  analysis_version: AnalysisVersion;
  chords?: Chord[];
}

export interface Section {
  label: string;
  start_sec: number;
  end_sec: number;
}

export interface StemAnalysis {
  audio_key: string | null;
  regions: StemRegion[];
  hits?: DrumHits;
}

export interface StemRegion {
  start_sec: number;
  end_sec: number;
  envelope: [number, number][]; // [t_relative_sec, energy_0_1]
  sub_label?: SubLabel;
  sub_label_confidence?: number;
}
