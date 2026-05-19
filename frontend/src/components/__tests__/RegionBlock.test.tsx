import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { RegionBlock } from '../RegionBlock';

describe('RegionBlock', () => {
  const region = {
    start_sec: 10,
    end_sec: 20,
    envelope: [[0, 0.5], [1, 0.8]] as [number, number][],
  };

  it('positions itself proportionally to duration on timeline', () => {
    const { container } = render(
      <RegionBlock
        region={region}
        durationSec={100}
        timelineWidth={1000}
        color="#6366f1"
        onClick={() => {}}
      />
    );
    const el = container.firstChild as HTMLElement;
    expect(el.style.left).toBe('100px'); // 10/100 * 1000
    expect(el.style.width).toBe('100px'); // (20-10)/100 * 1000
  });

  it('invokes onClick with region.start_sec', () => {
    const onClick = vi.fn();
    const { container } = render(
      <RegionBlock
        region={region}
        durationSec={100}
        timelineWidth={1000}
        color="#6366f1"
        onClick={onClick}
      />
    );
    (container.firstChild as HTMLElement).click();
    expect(onClick).toHaveBeenCalledWith(10);
  });
});
