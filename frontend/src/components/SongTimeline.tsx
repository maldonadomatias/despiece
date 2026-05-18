import { useState } from 'react';
import { AnalysisResult } from '@/types/song';
import { StemRow } from './StemRow';
import { SectionBar } from './SectionBar';
import { TimeAxis } from './TimeAxis';

const STEM_ORDER = ['vocals', 'drums', 'bass', 'other'];
const ROW_HEIGHT = 64;
const SECTION_HEIGHT = 28;
const AXIS_HEIGHT = 24;
const LABEL_WIDTH = 72;

const STEM_COLORS: Record<string, string> = {
  vocals: '#6366f1',
  drums: '#f59e0b',
  bass: '#10b981',
  other: '#8b5cf6',
};

interface Props {
  analysis: AnalysisResult;
}

export function SongTimeline({ analysis }: Props) {
  const [axisMode, setAxisMode] = useState<'seconds' | 'bars'>('bars');
  const [sectionLabels, setSectionLabels] = useState<Record<string, string>>({});

  const timelineWidth = Math.max(
    600,
    Math.min(1100, window.innerWidth - LABEL_WIDTH - 80)
  );

  function handleRename(label: string, name: string) {
    setSectionLabels((prev) => ({ ...prev, [label]: name }));
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm text-muted-foreground">Axis:</span>
        <button
          onClick={() => setAxisMode(axisMode === 'bars' ? 'seconds' : 'bars')}
          className="text-sm px-3 py-1 rounded border hover:bg-muted transition-colors"
        >
          {axisMode === 'bars' ? 'Bars' : 'Seconds'}
        </button>
        <span className="text-xs text-muted-foreground">
          Double-click a section to rename it.
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
            <div key={stemName} className="flex items-center mt-1">
              <div
                style={{ width: LABEL_WIDTH }}
                className="text-xs font-medium capitalize text-right pr-2 text-muted-foreground"
              >
                {stemName}
              </div>
              <div className="rounded overflow-hidden bg-muted/20">
                <StemRow
                  stemName={stemName}
                  data={analysis.stems[stemName]}
                  durationSec={analysis.duration_sec}
                  width={timelineWidth}
                  height={ROW_HEIGHT}
                />
              </div>
            </div>
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
