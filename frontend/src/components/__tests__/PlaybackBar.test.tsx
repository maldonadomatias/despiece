import { render, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PlaybackBar } from '../PlaybackBar';

describe('PlaybackBar', () => {
  it('renders a play button initially', () => {
    const audio = document.createElement('audio');
    const { getByRole } = render(
      <PlaybackBar audioRef={{ current: audio }} durationSec={60} timelineWidth={600} />
    );
    expect(getByRole('button', { name: /play/i })).toBeTruthy();
  });

  it('toggles to pause when clicked while playing', () => {
    const audio = document.createElement('audio');
    audio.play = vi.fn().mockResolvedValue(undefined);
    audio.pause = vi.fn();
    const { getByRole } = render(
      <PlaybackBar audioRef={{ current: audio }} durationSec={60} timelineWidth={600} />
    );
    fireEvent.click(getByRole('button', { name: /play/i }));
    act(() => {
      audio.dispatchEvent(new Event('play'));
    });
    expect(getByRole('button', { name: /pause/i })).toBeTruthy();
  });
});
