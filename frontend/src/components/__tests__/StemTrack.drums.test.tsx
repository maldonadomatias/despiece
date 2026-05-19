import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { StemTrack } from '../StemTrack';
import type { DrumHits, StemAnalysis, SubLabel } from '@/types/song';

const DRUM_HIT_COLORS: Record<'kick' | 'snare' | 'hihat' | 'cymbal' | 'unknown', string> = {
  kick: '#dc2626', snare: '#facc15', hihat: '#22d3ee', cymbal: '#a3e635', unknown: '#64748b',
};

function stem(hits?: DrumHits): StemAnalysis {
  return { audio_key: 'k', regions: [], hits };
}

const baseProps = {
  durationSec: 60,
  timelineWidth: 600,
  color: '#f59e0b',
  dim: false,
  soloed: false,
  muted: false,
  onToggleSolo: () => {},
  onToggleMute: () => {},
  onRegionClick: () => {},
  subLabelColors: undefined as Record<SubLabel, string> | undefined,
  drumHitColors: DRUM_HIT_COLORS,
};

describe('StemTrack drums sub-rows', () => {
  it('drums stem with no hits → no sub-rows', () => {
    render(<StemTrack name="drums" stem={stem()} {...baseProps} />);
    expect(screen.queryByText('kick')).not.toBeInTheDocument();
  });

  it('drums stem with 4 classes (no unknown) → 4 sub-rows render', () => {
    const hits: DrumHits = {
      kick:    [{ t_sec: 1, velocity: 0.8, confidence: 0.7 }],
      snare:   [{ t_sec: 2, velocity: 0.6, confidence: 0.6 }],
      hihat:   [{ t_sec: 3, velocity: 0.5, confidence: 0.5 }],
      cymbal:  [{ t_sec: 4, velocity: 0.9, confidence: 0.7 }],
      unknown: [],
    };
    render(<StemTrack name="drums" stem={stem(hits)} {...baseProps} />);
    expect(screen.getByText('kick')).toBeInTheDocument();
    expect(screen.getByText('snare')).toBeInTheDocument();
    expect(screen.getByText('hihat')).toBeInTheDocument();
    expect(screen.getByText('cymbal')).toBeInTheDocument();
    expect(screen.queryByText('unknown')).not.toBeInTheDocument();
  });

  it('drums stem with unknown hits → 5 sub-rows render', () => {
    const hits: DrumHits = {
      kick: [], snare: [], hihat: [], cymbal: [],
      unknown: [{ t_sec: 1, velocity: 0.5, confidence: 0.2 }],
    };
    render(<StemTrack name="drums" stem={stem(hits)} {...baseProps} />);
    expect(screen.getByText('unknown')).toBeInTheDocument();
  });

  it('vocals stem ignores hits prop (defensive)', () => {
    const hits: DrumHits = {
      kick:    [{ t_sec: 1, velocity: 0.8, confidence: 0.7 }],
      snare: [], hihat: [], cymbal: [], unknown: [],
    };
    render(<StemTrack name="vocals" stem={stem(hits)} {...baseProps} />);
    expect(screen.queryByText('kick')).not.toBeInTheDocument();
  });
});
