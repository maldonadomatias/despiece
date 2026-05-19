import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DrumHitRow } from '../DrumHitRow';
import type { DrumHit } from '@/types/song';

const hits: DrumHit[] = [
  { t_sec: 10, velocity: 0.5, confidence: 0.7 },
  { t_sec: 30, velocity: 1.0, confidence: 0.9 },
];

describe('DrumHitRow', () => {
  it('renders one tick per hit', () => {
    render(
      <DrumHitRow
        hits={hits}
        durationSec={60}
        timelineWidth={600}
        color="#dc2626"
        onHitClick={() => {}}
        label="kick"
      />
    );
    const ticks = screen.getAllByTestId('drum-hit-tick');
    expect(ticks).toHaveLength(2);
  });

  it('positions ticks proportionally to t_sec / durationSec', () => {
    render(
      <DrumHitRow
        hits={hits}
        durationSec={60}
        timelineWidth={600}
        color="#dc2626"
        onHitClick={() => {}}
        label="kick"
      />
    );
    const ticks = screen.getAllByTestId('drum-hit-tick');
    expect(ticks[0]).toHaveStyle({ left: '100px' });
    expect(ticks[1]).toHaveStyle({ left: '300px' });
  });

  it('applies opacity = 0.4 + 0.6 * velocity', () => {
    render(
      <DrumHitRow
        hits={hits}
        durationSec={60}
        timelineWidth={600}
        color="#dc2626"
        onHitClick={() => {}}
        label="kick"
      />
    );
    const ticks = screen.getAllByTestId('drum-hit-tick');
    expect(ticks[0]).toHaveStyle({ opacity: '0.7' });
    expect(ticks[1]).toHaveStyle({ opacity: '1' });
  });

  it('click on tick fires onHitClick with t_sec', () => {
    const spy = vi.fn();
    render(
      <DrumHitRow
        hits={hits}
        durationSec={60}
        timelineWidth={600}
        color="#dc2626"
        onHitClick={spy}
        label="kick"
      />
    );
    const ticks = screen.getAllByTestId('drum-hit-tick');
    ticks[1].click();
    expect(spy).toHaveBeenCalledWith(30);
  });

  it('isUnknown adds dashed border to ticks', () => {
    render(
      <DrumHitRow
        hits={hits}
        durationSec={60}
        timelineWidth={600}
        color="#64748b"
        onHitClick={() => {}}
        label="unknown"
        isUnknown
      />
    );
    const ticks = screen.getAllByTestId('drum-hit-tick');
    expect(ticks[0]).toHaveStyle({ borderStyle: 'dashed' });
  });
});
