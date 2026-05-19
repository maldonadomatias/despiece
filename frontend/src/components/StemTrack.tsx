import { StemAnalysis, SubLabel } from '@/types/song';
import { RegionBlock } from './RegionBlock';
import { StemControls } from './StemControls';

interface Props {
  name: string;
  stem: StemAnalysis;
  durationSec: number;
  timelineWidth: number;
  color: string;
  dim: boolean;
  soloed: boolean;
  muted: boolean;
  onToggleSolo: () => void;
  onToggleMute: () => void;
  onRegionClick: (startSec: number) => void;
  labelWidth?: number;
  rowHeight?: number;
  subLabelColors?: Record<SubLabel, string>;
}

export function StemTrack({
  name,
  stem,
  durationSec,
  timelineWidth,
  color,
  dim,
  soloed,
  muted,
  onToggleSolo,
  onToggleMute,
  onRegionClick,
  labelWidth = 96,
  rowHeight = 48,
  subLabelColors,
}: Props) {
  return (
    <div
      data-stem-track={name}
      className="flex items-center mt-1"
      style={{ opacity: dim ? 0.3 : 1, transition: 'opacity 120ms' }}
    >
      <div
        style={{ width: labelWidth }}
        className="text-xs font-medium capitalize text-right pr-2 text-muted-foreground flex items-center justify-end gap-2"
      >
        <StemControls
          soloed={soloed}
          muted={muted}
          onToggleSolo={onToggleSolo}
          onToggleMute={onToggleMute}
        />
        <span>{name}</span>
      </div>
      <div
        className="relative bg-muted/20 rounded overflow-hidden"
        style={{ width: timelineWidth, height: rowHeight }}
      >
        {stem.regions.map((region, i) => {
          const sub = name === 'other' ? region.sub_label : undefined;
          const subColor = sub && subLabelColors ? subLabelColors[sub] : undefined;
          return (
            <RegionBlock
              key={i}
              region={region}
              durationSec={durationSec}
              timelineWidth={timelineWidth}
              color={color}
              onClick={onRegionClick}
              rowHeight={rowHeight}
              subLabel={sub}
              subLabelColor={subColor}
            />
          );
        })}
      </div>
    </div>
  );
}
