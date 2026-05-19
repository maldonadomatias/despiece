import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { RegionBlock } from '../RegionBlock';

const baseRegion = {
  start_sec: 10,
  end_sec: 30,
  envelope: [[0, 0.5], [1, 0.8]] as [number, number][],
};

describe('RegionBlock sub-label', () => {
  it('renders label text when subLabel + subLabelColor passed and block is wide', () => {
    render(
      <RegionBlock
        region={{ ...baseRegion, sub_label: 'pad', sub_label_confidence: 0.78 }}
        durationSec={60}
        timelineWidth={1200}
        color="#94a3b8"
        subLabel="pad"
        subLabelColor="#8b5cf6"
        onClick={() => {}}
      />
    );
    expect(screen.getByText('pad')).toBeInTheDocument();
  });

  it('low confidence (<0.4) styles label as italic', () => {
    render(
      <RegionBlock
        region={{ ...baseRegion, sub_label: 'pad', sub_label_confidence: 0.3 }}
        durationSec={60}
        timelineWidth={1200}
        color="#94a3b8"
        subLabel="pad"
        subLabelColor="#8b5cf6"
        onClick={() => {}}
      />
    );
    const label = screen.getByText('pad');
    expect(label).toHaveStyle({ fontStyle: 'italic' });
  });

  it('does not render label text when subLabel is absent', () => {
    render(
      <RegionBlock
        region={baseRegion}
        durationSec={60}
        timelineWidth={1200}
        color="#94a3b8"
        onClick={() => {}}
      />
    );
    expect(screen.queryByText('pad')).not.toBeInTheDocument();
  });
});
