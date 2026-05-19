import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { StemTrack } from '../StemTrack';
import { StemAnalysis } from '@/types/song';

describe('StemTrack', () => {
  const stem: StemAnalysis = {
    audio_key: 'songs/abc/stems/vocals.mp3',
    regions: [
      { start_sec: 5, end_sec: 15, envelope: [[0, 0.5]] },
      { start_sec: 20, end_sec: 30, envelope: [[0, 0.3]] },
    ],
  };

  it('renders one block per region', () => {
    const { container } = render(
      <StemTrack
        name="vocals"
        stem={stem}
        durationSec={60}
        timelineWidth={600}
        color="#6366f1"
        dim={false}
        soloed={false}
        muted={false}
        onToggleSolo={() => {}}
        onToggleMute={() => {}}
        onRegionClick={() => {}}
      />
    );
    const blocks = container.querySelectorAll('canvas');
    expect(blocks.length).toBe(2);
  });

  it('dims container when dim=true', () => {
    const { container } = render(
      <StemTrack
        name="vocals"
        stem={stem}
        durationSec={60}
        timelineWidth={600}
        color="#6366f1"
        dim
        soloed={false}
        muted={false}
        onToggleSolo={() => {}}
        onToggleMute={() => {}}
        onRegionClick={() => {}}
      />
    );
    const row = container.querySelector('[data-stem-track]') as HTMLElement;
    expect(row.style.opacity).toBe('0.3');
  });

  it('passes start_sec into onRegionClick on click', () => {
    const handler = vi.fn();
    const { container } = render(
      <StemTrack
        name="vocals"
        stem={stem}
        durationSec={60}
        timelineWidth={600}
        color="#6366f1"
        dim={false}
        soloed={false}
        muted={false}
        onToggleSolo={() => {}}
        onToggleMute={() => {}}
        onRegionClick={handler}
      />
    );
    const block = container.querySelectorAll('[style*="position: absolute"]')[0] as HTMLElement;
    block.click();
    expect(handler).toHaveBeenCalledWith(5);
  });
});
