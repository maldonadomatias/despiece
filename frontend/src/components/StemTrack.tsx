import { DrumHitClass, DrumHits, StemAnalysis, SubLabel } from '@/types/song';
import { RegionBlock } from './RegionBlock';
import { StemControls } from './StemControls';
import { DrumHitRow } from './DrumHitRow';

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
  drumHitColors?: Record<DrumHitClass, string>;
}

const DRUM_SUB_ORDER: DrumHitClass[] = ['kick', 'snare', 'hihat', 'cymbal', 'unknown'];

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
  drumHitColors,
}: Props) {
  const drumSubRows: DrumHitClass[] =
    name === 'drums' && stem.hits
      ? DRUM_SUB_ORDER.filter((cls) =>
          cls === 'unknown'
            ? (stem.hits as DrumHits).unknown.length > 0
            : true
        )
      : [];

  return (
    <div
      data-stem-track={name}
      style={{ opacity: dim ? 0.3 : 1, transition: 'opacity 120ms' }}
    >
      <div className="flex items-center mt-1">
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

      {drumSubRows.map((cls) => (
        <DrumHitRow
          key={cls}
          hits={(stem.hits as DrumHits)[cls]}
          durationSec={durationSec}
          timelineWidth={timelineWidth}
          color={drumHitColors ? drumHitColors[cls] : color}
          onHitClick={onRegionClick}
          label={cls}
          isUnknown={cls === 'unknown'}
          labelWidth={labelWidth}
        />
      ))}
    </div>
  );
}
