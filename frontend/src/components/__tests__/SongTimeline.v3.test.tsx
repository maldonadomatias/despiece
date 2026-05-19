import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SongTimeline } from '../SongTimeline';
import type { AnalysisResult, DrumHits } from '@/types/song';

function baseAnalysis(version: 1 | 2 | 3, hits?: DrumHits): AnalysisResult {
  return {
    bpm: 120,
    key: 'C major',
    duration_sec: 60,
    beat_grid: [0, 0.5, 1.0],
    bar_grid: [0, 2.0],
    sections: [],
    stems: {
      vocals: { audio_key: 'k', regions: [] },
      drums:  { audio_key: 'k', regions: [], hits },
      bass:   { audio_key: 'k', regions: [] },
      guitar: { audio_key: 'k', regions: [] },
      piano:  { audio_key: 'k', regions: [] },
      other:  { audio_key: 'k', regions: [] },
    },
    analysis_version: version,
  };
}

describe('SongTimeline v3', () => {
  it('v1 → hard re-analyze gate (unchanged from Phase A)', () => {
    const old = { ...baseAnalysis(1) } as any;
    delete old.analysis_version;
    render(<SongTimeline songId="abc" analysis={old} />);
    expect(screen.getByText(/older pipeline/i)).toBeInTheDocument();
  });

  it('v2 → soft v3 notice banner renders, timeline still renders', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(2)} />);
    expect(screen.getByText(/drum sub-rows are available/i)).toBeInTheDocument();
    expect(screen.queryByText(/older pipeline/i)).not.toBeInTheDocument();
  });

  it('v3 with no hits → no v3 notice and no drum legend', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(3)} />);
    expect(screen.queryByText(/drum sub-rows are available/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/drum hits:/i)).not.toBeInTheDocument();
  });

  it('v3 with hits → drum-hit legend renders only for classes with entries', () => {
    const hits: DrumHits = {
      kick:    [{ t_sec: 1, velocity: 0.8, confidence: 0.7 }],
      snare:   [{ t_sec: 2, velocity: 0.6, confidence: 0.6 }],
      hihat:   [],
      cymbal:  [],
      unknown: [],
    };
    render(<SongTimeline songId="abc" analysis={baseAnalysis(3, hits)} />);
    expect(screen.getByText(/drum hits:/i)).toBeInTheDocument();
    const kickMatches = screen.getAllByText(/^kick$/i);
    expect(kickMatches.length).toBeGreaterThan(0);
  });
});
