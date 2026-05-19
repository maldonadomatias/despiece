import { useEffect, useRef } from 'react';
import { StemRegion } from '@/types/song';

interface Props {
  region: StemRegion;
  durationSec: number;
  timelineWidth: number;
  color: string;
  onClick: (startSec: number) => void;
  rowHeight?: number;
}

export function RegionBlock({
  region,
  durationSec,
  timelineWidth,
  color,
  onClick,
  rowHeight = 48,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const left = (region.start_sec / durationSec) * timelineWidth;
  const width = ((region.end_sec - region.start_sec) / durationSec) * timelineWidth;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(rowHeight * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = color + '22'; // ~13% alpha background
    ctx.fillRect(0, 0, width, rowHeight);

    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, rowHeight - 1);

    if (region.envelope.length === 0) return;
    const regionDur = region.end_sec - region.start_sec;
    ctx.fillStyle = color;
    for (const [t, v] of region.envelope) {
      const x = (t / regionDur) * width;
      const h = Math.max(1, v * (rowHeight - 4));
      ctx.fillRect(x, rowHeight - h - 2, 1, h);
    }
  }, [region, width, rowHeight, color]);

  return (
    <div
      style={{
        position: 'absolute',
        left: `${left}px`,
        width: `${width}px`,
        height: `${rowHeight}px`,
        top: 0,
        cursor: 'pointer',
      }}
      onClick={() => onClick(region.start_sec)}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  );
}
