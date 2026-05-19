import { useMemo, useRef, useState } from 'react';
import { AnalysisResult } from '@/types/song';
import { StemTrack } from './StemTrack';
import { SectionBar } from './SectionBar';
import { TimeAxis } from './TimeAxis';
import { PlaybackBar } from './PlaybackBar';
import { getMixAudioUrl } from '@/lib/api';

const STEM_ORDER = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];

const STEM_COLORS: Record<string, string> = {
  vocals: '#6366f1',
  drums:  '#f59e0b',
  bass:   '#10b981',
  guitar: '#a78bfa',
  piano:  '#06b6d4',
  other:  '#94a3b8',
};

const ROW_HEIGHT = 48;
const SECTION_HEIGHT = 28;
const AXIS_HEIGHT = 24;
const LABEL_WIDTH = 96;

interface Props {
  songId: string;
  analysis: AnalysisResult;
}

export function SongTimeline({ songId, analysis }: Props) {
  const [axisMode, setAxisMode] = useState<'seconds' | 'bars'>('bars');
  const [sectionLabels, setSectionLabels] = useState<Record<string, string>>({});
  const [soloed, setSoloed] = useState<Set<string>>(new Set());
  const [muted, setMuted] = useState<Set<string>>(new Set());
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const timelineWidth = useMemo(
    () => Math.max(600, Math.min(1400, window.innerWidth - LABEL_WIDTH - 80)),
    []
  );

  function toggleSolo(name: string) {
    setSoloed((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function toggleMute(name: string) {
    setMuted((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function isDim(name: string) {
    if (muted.has(name)) return true;
    if (soloed.size > 0 && !soloed.has(name)) return true;
    return false;
  }

  function handleRegionClick(startSec: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = startSec;
    audio.play();
  }

  function handleRename(label: string, name: string) {
    setSectionLabels((prev) => ({ ...prev, [label]: name }));
  }

  return (
    <div className="space-y-4">
      <audio ref={audioRef} src={getMixAudioUrl(songId)} preload="auto" />

      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm text-muted-foreground">Axis:</span>
        <button
          onClick={() => setAxisMode(axisMode === 'bars' ? 'seconds' : 'bars')}
          className="text-sm px-3 py-1 rounded border hover:bg-muted transition-colors"
        >
          {axisMode === 'bars' ? 'Bars' : 'Seconds'}
        </button>
        <span className="text-xs text-muted-foreground">
          Double-click a section to rename it. Click a region to jump audio.
        </span>
      </div>

      <div className="overflow-x-auto">
        <div style={{ minWidth: LABEL_WIDTH + timelineWidth }}>
          <div className="flex items-center">
            <div
              style={{ width: LABEL_WIDTH }}
              className="text-xs text-muted-foreground pr-2 text-right"
            >
              Sections
            </div>
            <SectionBar
              sections={analysis.sections}
              durationSec={analysis.duration_sec}
              width={timelineWidth}
              height={SECTION_HEIGHT}
              labels={sectionLabels}
              onRename={handleRename}
            />
          </div>

          {STEM_ORDER.filter((s) => analysis.stems[s]).map((stemName) => (
            <StemTrack
              key={stemName}
              name={stemName}
              stem={analysis.stems[stemName]}
              durationSec={analysis.duration_sec}
              timelineWidth={timelineWidth}
              color={STEM_COLORS[stemName] ?? '#94a3b8'}
              dim={isDim(stemName)}
              soloed={soloed.has(stemName)}
              muted={muted.has(stemName)}
              onToggleSolo={() => toggleSolo(stemName)}
              onToggleMute={() => toggleMute(stemName)}
              onRegionClick={handleRegionClick}
              labelWidth={LABEL_WIDTH}
              rowHeight={ROW_HEIGHT}
            />
          ))}

          <div className="flex items-center mt-1">
            <div style={{ width: LABEL_WIDTH }} />
            <TimeAxis
              durationSec={analysis.duration_sec}
              bpm={analysis.bpm}
              width={timelineWidth}
              height={AXIS_HEIGHT}
              mode={axisMode}
            />
          </div>

          <PlaybackBar
            audioRef={audioRef}
            durationSec={analysis.duration_sec}
            timelineWidth={timelineWidth}
            labelWidth={LABEL_WIDTH}
          />
        </div>
      </div>

      <div className="flex gap-4 flex-wrap text-xs text-muted-foreground">
        {STEM_ORDER.map((s) => (
          <span key={s} className="flex items-center gap-1">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{ background: STEM_COLORS[s] }}
            />
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
