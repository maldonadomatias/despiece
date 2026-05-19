import { useMemo, useState } from 'react';
import { AnalysisResult, Chord, ChordLabel, DrumHitClass, DrumHits, SubLabel } from '@/types/song';
import { StemTrack } from './StemTrack';
import { SectionBar } from './SectionBar';
import { TimeAxis } from './TimeAxis';
import { PlaybackBar } from './PlaybackBar';
import { ChordBar } from './ChordBar';
import { getStemAudioUrl } from '@/lib/api';
import { audible } from '@/lib/audible';
import { useStemTransport } from '@/hooks/useStemTransport';

const STEM_ORDER = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];

const STEM_COLORS: Record<string, string> = {
  vocals: '#6366f1',
  drums:  '#f59e0b',
  bass:   '#10b981',
  guitar: '#a78bfa',
  piano:  '#06b6d4',
  other:  '#94a3b8',
};

const SUB_LABEL_COLORS: Record<SubLabel, string> = {
  lead:       '#ef4444',
  pad:        '#8b5cf6',
  synth:      '#ec4899',
  strings:    '#f97316',
  fx:         '#14b8a6',
  other_misc: '#94a3b8',
};

const DRUM_HIT_COLORS: Record<DrumHitClass, string> = {
  kick:    '#dc2626',
  snare:   '#facc15',
  hihat:   '#22d3ee',
  cymbal:  '#a3e635',
  unknown: '#64748b',
};

const CHORD_ROOT_COLORS: Record<string, string> = {
  C:    '#ef4444',
  'C#': '#f97316',
  D:    '#f59e0b',
  'D#': '#eab308',
  E:    '#84cc16',
  F:    '#22c55e',
  'F#': '#14b8a6',
  G:    '#06b6d4',
  'G#': '#3b82f6',
  A:    '#8b5cf6',
  'A#': '#a855f7',
  B:    '#ec4899',
};
const NO_CHORD_COLOR = '#475569';
const CHORD_HEIGHT = 28;

const ROW_HEIGHT = 48;
const SECTION_HEIGHT = 28;
const AXIS_HEIGHT = 24;
const LABEL_WIDTH = 96;
const SUPPORTED_ANALYSIS_VERSION = 2;
const PREFERRED_ANALYSIS_VERSION = 4;

interface Props {
  songId: string;
  analysis: AnalysisResult;
}

export function SongTimeline({ songId, analysis }: Props) {
  const [axisMode, setAxisMode] = useState<'seconds' | 'bars'>('bars');
  const [sectionLabels, setSectionLabels] = useState<Record<string, string>>({});
  const [soloed, setSoloed] = useState<Set<string>>(new Set());
  const [muted, setMuted] = useState<Set<string>>(new Set());
  const transport = useStemTransport();

  const timelineWidth = useMemo(
    () => Math.max(600, Math.min(1400, window.innerWidth - LABEL_WIDTH - 80)),
    []
  );

  const activeStems = useMemo(
    () =>
      STEM_ORDER.filter(
        (s) => analysis.stems[s] && analysis.stems[s].audio_key !== null
      ),
    [analysis.stems]
  );

  const otherSubLabels = useMemo<SubLabel[]>(() => {
    const other = analysis.stems['other'];
    if (!other) return [];
    const set = new Set<SubLabel>();
    for (const r of other.regions) {
      if (r.sub_label) set.add(r.sub_label);
    }
    return Array.from(set);
  }, [analysis.stems]);

  const activeDrumHitClasses = useMemo<DrumHitClass[]>(() => {
    const drums = analysis.stems['drums'];
    const hits: DrumHits | undefined = drums?.hits;
    if (!hits) return [];
    const order: DrumHitClass[] = ['kick', 'snare', 'hihat', 'cymbal', 'unknown'];
    return order.filter((cls) => hits[cls].length > 0);
  }, [analysis.stems]);

  const version = analysis.analysis_version ?? 1;
  if (version < SUPPORTED_ANALYSIS_VERSION) {
    return (
      <div className="p-6 border rounded bg-muted/20 text-sm">
        <p className="font-medium mb-1">This song was analyzed with an older pipeline.</p>
        <p className="text-muted-foreground">
          Re-analyze it to get pro-grade stems and "other"-stem sub-labels.
        </p>
      </div>
    );
  }
  const showV3Notice = version < PREFERRED_ANALYSIS_VERSION;

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
    return !audible(name, soloed, muted);
  }

  function handleRegionClick(startSec: number) {
    transport.playFrom(startSec);
  }

  function handleRename(label: string, name: string) {
    setSectionLabels((prev) => ({ ...prev, [label]: name }));
  }

  return (
    <div className="space-y-4">
      {showV3Notice && (
        <div className="p-3 border rounded bg-amber-50 text-xs text-amber-900">
          Newer analysis features available — re-analyze for drum sub-rows + chord track.
        </div>
      )}

      {activeStems.map((name) => (
        <audio
          key={name}
          ref={(el) => transport.registerAudio(name, el)}
          src={getStemAudioUrl(songId, name)}
          muted={!audible(name, soloed, muted)}
          preload="auto"
        />
      ))}

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

          {analysis.chords && analysis.chords.length > 0 && (
            <div className="flex items-center mt-1">
              <div
                style={{ width: LABEL_WIDTH }}
                className="text-xs text-muted-foreground pr-2 text-right"
              >
                Chords
              </div>
              <ChordBar
                chords={analysis.chords}
                durationSec={analysis.duration_sec}
                width={timelineWidth}
                height={CHORD_HEIGHT}
                rootColors={CHORD_ROOT_COLORS}
                noChordColor={NO_CHORD_COLOR}
                onChordClick={handleRegionClick}
              />
            </div>
          )}

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
              subLabelColors={SUB_LABEL_COLORS}
              drumHitColors={DRUM_HIT_COLORS}
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
            transport={transport}
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

      {otherSubLabels.length > 0 && (
        <div className="flex gap-4 flex-wrap text-xs text-muted-foreground">
          <span className="font-medium">Sub-labels:</span>
          {otherSubLabels.map((sub) => (
            <span key={sub} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ background: SUB_LABEL_COLORS[sub] }}
              />
              {sub}
            </span>
          ))}
        </div>
      )}

      {activeDrumHitClasses.length > 0 && (
        <div className="flex gap-4 flex-wrap text-xs text-muted-foreground">
          <span className="font-medium">Drum hits:</span>
          {activeDrumHitClasses.map((cls) => (
            <span key={cls} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ background: DRUM_HIT_COLORS[cls] }}
              />
              {cls}
            </span>
          ))}
        </div>
      )}

      {analysis.chords && analysis.chords.length > 0 && (
        <div className="flex gap-3 flex-wrap text-xs text-muted-foreground">
          <span className="font-medium">Chord roots:</span>
          {(['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'] as const).map((root) => (
            <span key={root} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ background: CHORD_ROOT_COLORS[root] }}
              />
              {root}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
