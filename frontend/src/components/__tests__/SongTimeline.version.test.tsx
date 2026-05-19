import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SongTimeline } from '../SongTimeline';
import type { AnalysisResult } from '@/types/song';

function baseAnalysis(version: number, otherRegion: any = null): AnalysisResult {
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
      other:  { audio_key: 'k', regions: otherRegion ? [otherRegion] : [] },
    },
    analysis_version: version as 1 | 2,
  };
}

describe('SongTimeline', () => {
  it('shows "Re-analyze" notice when analysis_version is missing or < 2', () => {
    const old = { ...baseAnalysis(1) } as any;
    delete old.analysis_version;
    render(<SongTimeline songId="abc" analysis={old} />);
    expect(screen.getByText(/older pipeline/i)).toBeInTheDocument();
  });

  it('renders timeline when analysis_version >= 2', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(2)} />);
    expect(screen.queryByText(/older pipeline/i)).not.toBeInTheDocument();
  });

  it('renders the sub-label legend when at least one other region has sub_label', () => {
    const analysis = baseAnalysis(2, {
      start_sec: 0,
      end_sec: 10,
      envelope: [],
      sub_label: 'pad',
      sub_label_confidence: 0.8,
    });
    render(<SongTimeline songId="abc" analysis={analysis} />);
    expect(screen.getByText(/sub-labels/i)).toBeInTheDocument();
    expect(screen.getAllByText(/pad/i).length).toBeGreaterThan(0);
  });

  it('hides the sub-label legend when no other region has sub_label', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(2)} />);
    expect(screen.queryByText(/sub-labels/i)).not.toBeInTheDocument();
  });
});
