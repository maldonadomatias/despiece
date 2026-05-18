import { StemData } from '@/types/song';

const STEM_COLORS: Record<string, string> = {
  vocals: '#6366f1',
  drums: '#f59e0b',
  bass: '#10b981',
  other: '#8b5cf6',
};

interface Props {
  stemName: string;
  data: StemData;
  durationSec: number;
  width: number;
  height: number;
}

export function StemRow({ stemName, data, durationSec, width, height }: Props) {
  const color = STEM_COLORS[stemName] ?? '#94a3b8';
  const envelope = data.envelope;

  if (envelope.length === 0) return null;

  const scaleX = (t: number) => (t / durationSec) * width;
  const scaleY = (e: number) => height - e * height;

  const pathD =
    `M 0,${height} ` +
    envelope
      .map(([t, e]) => `L ${scaleX(t).toFixed(1)},${scaleY(e).toFixed(1)}`)
      .join(' ') +
    ` L ${width},${height} Z`;

  return (
    <svg width={width} height={height} className="block">
      <path d={pathD} fill={color} opacity={0.75} />
    </svg>
  );
}
