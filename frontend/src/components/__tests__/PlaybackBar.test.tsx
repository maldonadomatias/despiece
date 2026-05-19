import { render, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PlaybackBar } from '../PlaybackBar';
import type { StemTransport } from '@/hooks/useStemTransport';

function makeTransport(overrides: Partial<StemTransport> = {}): StemTransport {
  return {
    registerAudio: vi.fn(),
    toggle: vi.fn(),
    seek: vi.fn(),
    playFrom: vi.fn(),
    playing: false,
    currentTime: 0,
    ...overrides,
  };
}

describe('PlaybackBar', () => {
  it('renders Play button when not playing', () => {
    const { getByRole } = render(
      <PlaybackBar transport={makeTransport()} durationSec={60} timelineWidth={600} />
    );
    expect(getByRole('button', { name: /play/i })).toBeTruthy();
  });

  it('renders Pause button when playing', () => {
    const { getByRole } = render(
      <PlaybackBar
        transport={makeTransport({ playing: true })}
        durationSec={60}
        timelineWidth={600}
      />
    );
    expect(getByRole('button', { name: /pause/i })).toBeTruthy();
  });

  it('calls transport.toggle when play/pause clicked', () => {
    const toggle = vi.fn();
    const { getByRole } = render(
      <PlaybackBar
        transport={makeTransport({ toggle })}
        durationSec={60}
        timelineWidth={600}
      />
    );
    fireEvent.click(getByRole('button', { name: /play/i }));
    expect(toggle).toHaveBeenCalled();
  });
});
