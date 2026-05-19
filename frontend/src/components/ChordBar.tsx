import { Chord, ChordLabel } from '@/types/song';

interface Props {
  chords: Chord[];
  durationSec: number;
  width: number;
  height?: number;
  rootColors: Record<string, string>;
  noChordColor: string;
  onChordClick?: (startSec: number) => void;
}

const NO_CHORD: ChordLabel = 'N';

function getRoot(label: ChordLabel): string {
  if (label === NO_CHORD) return NO_CHORD;
  return label.endsWith('m') ? label.slice(0, -1) : label;
}

function isMinor(label: ChordLabel): boolean {
  return label !== NO_CHORD && label.endsWith('m');
}

export function ChordBar({
  chords,
  durationSec,
  width,
  height = 28,
  rootColors,
  noChordColor,
  onChordClick,
}: Props) {
  return (
    <div
      className="relative bg-muted/20 rounded overflow-hidden"
      style={{ width, height }}
    >
      {chords.map((chord, i) => {
        const left = (chord.start_sec / durationSec) * width;
        const blockWidth = ((chord.end_sec - chord.start_sec) / durationSec) * width;
        const root = getRoot(chord.label);
        const isNoChord = chord.label === NO_CHORD;
        const background = isNoChord ? noChordColor : rootColors[root] ?? noChordColor;
        const minor = isMinor(chord.label);
        const startWhole = Math.round(chord.start_sec);
        const endWhole = Math.round(chord.end_sec);
        return (
          <div
            key={i}
            data-testid="chord-block"
            onClick={() => onChordClick?.(chord.start_sec)}
            title={`${chord.label} · ${startWhole}s – ${endWhole}s`}
            style={{
              position: 'absolute',
              left: `${left}px`,
              top: 0,
              width: `${blockWidth}px`,
              height: '100%',
              background,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              fontWeight: 600,
              color: '#fff',
              textShadow: '0 0 2px rgba(0,0,0,0.6)',
              overflow: 'hidden',
              whiteSpace: 'nowrap',
            }}
          >
            {minor && (
              <div
                data-testid="chord-block-overlay"
                style={{
                  position: 'absolute',
                  inset: 0,
                  background: 'rgba(0, 0, 0, 0.3)',
                  pointerEvents: 'none',
                }}
              />
            )}
            {!isNoChord && (
              <span
                style={{
                  position: 'relative',
                  zIndex: 1,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  maxWidth: '100%',
                  padding: '0 4px',
                }}
              >
                {chord.label}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
