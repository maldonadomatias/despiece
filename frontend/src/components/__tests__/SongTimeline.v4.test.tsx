import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SongTimeline } from '../SongTimeline';
import type { AnalysisResult, Chord } from '@/types/song';

function baseAnalysis(version: 1 | 2 | 3 | 4, chords?: Chord[]): AnalysisResult {
  return {
    bpm: 120,
    key: 'C major',
    duration_sec: 60,
    beat_grid: [0, 0.5, 1.0],
    bar_grid: [0, 2.0],
    sections: [],
    stems: {
      vocals: { audio_key: 'k', regions: [] },
      drums:  { audio_key: 'k', regions: [] },
      bass:   { audio_key: 'k', regions: [] },
      guitar: { audio_key: 'k', regions: [] },
      piano:  { audio_key: 'k', regions: [] },
      other:  { audio_key: 'k', regions: [] },
    },
    analysis_version: version,
    chords,
  };
}

describe('SongTimeline v4', () => {
  it('v3 with no chords → soft notice mentions both drum sub-rows and chord track', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(3)} />);
    expect(
      screen.getByText(/newer analysis features available/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/drum sub-rows.*chord track/i)
    ).toBeInTheDocument();
  });

  it('v4 with chords renders chord row between sections and stems', () => {
    const chords: Chord[] = [
      { start_sec: 0, end_sec: 30, label: 'Am' },
      { start_sec: 30, end_sec: 60, label: 'F' },
    ];
    render(<SongTimeline songId="abc" analysis={baseAnalysis(4, chords)} />);
    const blocks = screen.getAllByTestId('chord-block');
    expect(blocks).toHaveLength(2);
    expect(screen.queryByText(/newer analysis features available/i)).not.toBeInTheDocument();
  });

  it('v4 with no chords → no chord row, no chord legend', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(4)} />);
    expect(screen.queryByTestId('chord-block')).not.toBeInTheDocument();
    expect(screen.queryByText(/chord roots:/i)).not.toBeInTheDocument();
  });

  it('chord legend renders only when chords array non-empty', () => {
    const chords: Chord[] = [{ start_sec: 0, end_sec: 60, label: 'C' }];
    render(<SongTimeline songId="abc" analysis={baseAnalysis(4, chords)} />);
    expect(screen.getByText(/chord roots:/i)).toBeInTheDocument();
  });
});
