import { render, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { StemControls } from '../StemControls';

describe('StemControls', () => {
  it('renders S and M buttons and fires callbacks', () => {
    const onSolo = vi.fn();
    const onMute = vi.fn();
    const { getByRole } = render(
      <StemControls soloed={false} muted={false} onToggleSolo={onSolo} onToggleMute={onMute} />
    );
    fireEvent.click(getByRole('button', { name: /solo/i }));
    fireEvent.click(getByRole('button', { name: /mute/i }));
    expect(onSolo).toHaveBeenCalled();
    expect(onMute).toHaveBeenCalled();
  });

  it('highlights soloed state visually', () => {
    const { getByRole } = render(
      <StemControls soloed muted={false} onToggleSolo={() => {}} onToggleMute={() => {}} />
    );
    expect(getByRole('button', { name: /solo/i }).getAttribute('aria-pressed')).toBe('true');
  });
});
