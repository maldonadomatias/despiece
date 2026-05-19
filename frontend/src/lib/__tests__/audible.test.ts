import { describe, it, expect } from 'vitest';
import { audible } from '../audible';

describe('audible', () => {
  const NONE = new Set<string>();

  it('returns true when no solo and no mute', () => {
    expect(audible('vocals', NONE, NONE)).toBe(true);
  });

  it('returns false when stem is muted', () => {
    expect(audible('vocals', NONE, new Set(['vocals']))).toBe(false);
  });

  it('returns true for soloed stem', () => {
    expect(audible('vocals', new Set(['vocals']), NONE)).toBe(true);
  });

  it('returns false for non-soloed stem when something is soloed', () => {
    expect(audible('drums', new Set(['vocals']), NONE)).toBe(false);
  });

  it('mute wins over solo', () => {
    expect(audible('vocals', new Set(['vocals']), new Set(['vocals']))).toBe(false);
  });

  it('supports multiple soloed stems', () => {
    const solo = new Set(['vocals', 'drums']);
    expect(audible('vocals', solo, NONE)).toBe(true);
    expect(audible('drums', solo, NONE)).toBe(true);
    expect(audible('bass', solo, NONE)).toBe(false);
  });
});
