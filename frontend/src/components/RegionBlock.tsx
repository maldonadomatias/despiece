import { useEffect, useRef } from 'react';
import { StemRegion, SubLabel } from '@/types/song';

interface Props {
  region: StemRegion;
  durationSec: number;
  timelineWidth: number;
  color: string;
  onClick: (startSec: number) => void;
  rowHeight?: number;
  subLabel?: SubLabel;
  subLabelColor?: string;
}

const LABEL_MIN_WIDTH_PX = 60;

export function RegionBlock({
  region,
  durationSec,
  timelineWidth,
  color,
  onClick,
  rowHeight = 48,
  subLabel,
  subLabelColor,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const left = (region.start_sec / durationSec) * timelineWidth;
  const width = ((region.end_sec - region.start_sec) / durationSec) * timelineWidth;
  const effectiveColor = subLabelColor ?? color;
  const confidence = region.sub_label_confidence ?? 1;
  const lowConfidence = confidence < 0.4;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(rowHeight * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = effectiveColor + '22';
    ctx.fillRect(0, 0, width, rowHeight);
    ctx.strokeStyle = effectiveColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, rowHeight - 1);

    if (region.envelope.length === 0) return;
    const regionDur = region.end_sec - region.start_sec;
    ctx.fillStyle = effectiveColor;
    for (const [t, v] of region.envelope) {
      const x = (t / regionDur) * width;
      const h = Math.max(1, v * (rowHeight - 4));
      ctx.fillRect(x, rowHeight - h - 2, 1, h);
    }
  }, [region, width, rowHeight, effectiveColor]);

  const showLabel = !!subLabel && width >= LABEL_MIN_WIDTH_PX;
  const titleText =
    subLabel && region.sub_label_confidence !== undefined
      ? `${subLabel} · ${Math.round(region.sub_label_confidence * 100)}%`
      : subLabel ?? undefined;

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
      title={titleText}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
      {showLabel && (
        <span
          style={{
            position: 'absolute',
            top: 2,
            left: 4,
            fontSize: 10,
            fontStyle: lowConfidence ? 'italic' : 'normal',
            opacity: lowConfidence ? 0.6 : 1,
            color: effectiveColor,
            pointerEvents: 'none',
            textShadow: '0 0 2px rgba(0,0,0,0.4)',
          }}
        >
          {subLabel}
        </span>
      )}
    </div>
  );
}
