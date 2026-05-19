import { StemTransport } from '@/hooks/useStemTransport';

interface Props {
  transport: StemTransport;
  durationSec: number;
  timelineWidth: number;
  labelWidth?: number;
}

export function PlaybackBar({ transport, durationSec, timelineWidth, labelWidth = 96 }: Props) {
  const { playing, currentTime, toggle, seek } = transport;
  const cursorPx = durationSec > 0 ? (currentTime / durationSec) * timelineWidth : 0;

  function onScrub(e: React.MouseEvent<HTMLDivElement>) {
    if (durationSec <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    seek((x / timelineWidth) * durationSec);
  }

  return (
    <div className="flex items-center mt-2">
      <div style={{ width: labelWidth }} className="pr-2 text-right">
        <button
          type="button"
          aria-label={playing ? 'pause' : 'play'}
          onClick={toggle}
          className="text-xs px-2 py-1 rounded border hover:bg-muted"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
      </div>
      <div
        className="relative bg-muted/30 rounded cursor-pointer"
        style={{ width: timelineWidth, height: 8 }}
        onClick={onScrub}
      >
        <div
          className="absolute top-0 bottom-0 bg-primary"
          style={{ left: 0, width: `${cursorPx}px` }}
        />
      </div>
    </div>
  );
}
