import { DrumHit } from '@/types/song';

interface Props {
  hits: DrumHit[];
  durationSec: number;
  timelineWidth: number;
  color: string;
  onHitClick: (tSec: number) => void;
  label: string;
  rowHeight?: number;
  isUnknown?: boolean;
  labelWidth?: number;
}

const TICK_WIDTH_PX = 2;

export function DrumHitRow({
  hits,
  durationSec,
  timelineWidth,
  color,
  onHitClick,
  label,
  rowHeight = 24,
  isUnknown = false,
  labelWidth = 96,
}: Props) {
  return (
    <div className="flex items-center mt-0.5">
      <div
        style={{ width: labelWidth, paddingLeft: 16 }}
        className="text-[10px] text-muted-foreground text-right pr-2"
      >
        {label}
      </div>
      <div
        className="relative bg-muted/10 rounded overflow-hidden"
        style={{ width: timelineWidth, height: rowHeight }}
      >
        {hits.map((hit, i) => {
          const left = (hit.t_sec / durationSec) * timelineWidth;
          const opacity = 0.4 + 0.6 * hit.velocity;
          return (
            <div
              key={i}
              data-testid="drum-hit-tick"
              onClick={() => onHitClick(hit.t_sec)}
              title={`${label} · vel ${Math.round(hit.velocity * 100)}% · conf ${Math.round(hit.confidence * 100)}%`}
              style={{
                position: 'absolute',
                left: `${left}px`,
                top: 0,
                width: `${TICK_WIDTH_PX}px`,
                height: '100%',
                background: color,
                opacity,
                cursor: 'pointer',
                borderStyle: isUnknown ? 'dashed' : 'solid',
                borderWidth: isUnknown ? '1px' : '0',
                borderColor: isUnknown ? color : 'transparent',
                boxSizing: 'border-box',
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
