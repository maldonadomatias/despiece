import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ChordBar } from '../ChordBar';
import type { Chord } from '@/types/song';

const chords: Chord[] = [
  { start_sec: 0,  end_sec: 8,  label: 'Am' },
  { start_sec: 8,  end_sec: 12, label: 'F'  },
  { start_sec: 12, end_sec: 16, label: 'C'  },
  { start_sec: 16, end_sec: 24, label: 'N'  },
];

const palette: Record<string, string> = {
  C: '#ef4444', 'C#': '#f97316', D: '#f59e0b', 'D#': '#eab308',
  E: '#84cc16', F: '#22c55e', 'F#': '#14b8a6', G: '#06b6d4',
  'G#': '#3b82f6', A: '#8b5cf6', 'A#': '#a855f7', B: '#ec4899',
};
const noChordColor = '#475569';

describe('ChordBar', () => {
  it('renders one block per chord', () => {
    render(
      <ChordBar
        chords={chords}
        durationSec={24}
        width={1200}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const blocks = screen.getAllByTestId('chord-block');
    expect(blocks).toHaveLength(4);
  });

  it('positions blocks by start_sec / durationSec', () => {
    render(
      <ChordBar
        chords={chords}
        durationSec={24}
        width={1200}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const blocks = screen.getAllByTestId('chord-block');
    expect(blocks[0]).toHaveStyle({ left: '0px',   width: '400px' });
    expect(blocks[1]).toHaveStyle({ left: '400px', width: '200px' });
    expect(blocks[2]).toHaveStyle({ left: '600px', width: '200px' });
    expect(blocks[3]).toHaveStyle({ left: '800px', width: '400px' });
  });

  it('major chord uses root color at full opacity (no overlay)', () => {
    render(
      <ChordBar
        chords={[{ start_sec: 0, end_sec: 4, label: 'C' }]}
        durationSec={4}
        width={400}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const block = screen.getByTestId('chord-block');
    expect(block).toHaveStyle({ background: 'rgb(239, 68, 68)' });
    expect(screen.queryByTestId('chord-block-overlay')).not.toBeInTheDocument();
  });

  it('minor chord uses root color + 30% black overlay', () => {
    render(
      <ChordBar
        chords={[{ start_sec: 0, end_sec: 4, label: 'Am' }]}
        durationSec={4}
        width={400}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const block = screen.getByTestId('chord-block');
    expect(block).toHaveStyle({ background: 'rgb(139, 92, 246)' });
    const overlay = screen.getByTestId('chord-block-overlay');
    expect(overlay).toBeInTheDocument();
  });

  it('"N" chord uses noChordColor and renders no label text', () => {
    render(
      <ChordBar
        chords={[{ start_sec: 0, end_sec: 4, label: 'N' }]}
        durationSec={4}
        width={400}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const block = screen.getByTestId('chord-block');
    expect(block).toHaveStyle({ background: 'rgb(71, 85, 105)' });
    expect(screen.queryByText('N')).not.toBeInTheDocument();
  });

  it('click on a block fires onChordClick with start_sec', () => {
    const spy = vi.fn();
    render(
      <ChordBar
        chords={chords}
        durationSec={24}
        width={1200}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={spy}
      />
    );
    const blocks = screen.getAllByTestId('chord-block');
    blocks[1].click();
    expect(spy).toHaveBeenCalledWith(8);
  });
});
