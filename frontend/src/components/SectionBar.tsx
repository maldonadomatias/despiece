import { useState } from 'react';
import { Section } from '@/types/song';

const SECTION_COLORS = [
  '#cbd5e1', '#94a3b8', '#64748b', '#475569', '#334155',
];

interface Props {
  sections: Section[];
  durationSec: number;
  width: number;
  height: number;
  labels: Record<string, string>;
  onRename: (label: string, name: string) => void;
}

export function SectionBar({
  sections,
  durationSec,
  width,
  height,
  labels,
  onRename,
}: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');

  if (sections.length === 0) {
    return <div style={{ width, height }} className="bg-muted/30 rounded" />;
  }

  function startEdit(label: string) {
    setEditing(label);
    setDraft(labels[label] ?? label);
  }

  function commitEdit() {
    if (editing) onRename(editing, draft.trim() || editing);
    setEditing(null);
  }

  return (
    <svg width={width} height={height} className="block">
      {sections.map((sec, i) => {
        const x = (sec.start_sec / durationSec) * width;
        const w = ((sec.end_sec - sec.start_sec) / durationSec) * width;
        const displayName = labels[sec.label] ?? sec.label;
        const isEditing = editing === sec.label;
        return (
          <g
            key={sec.label}
            onDoubleClick={() => startEdit(sec.label)}
            style={{ cursor: 'pointer' }}
          >
            <rect
              x={x}
              y={0}
              width={w}
              height={height}
              fill={SECTION_COLORS[i % SECTION_COLORS.length]}
              rx={2}
            />
            {w > 24 && !isEditing && (
              <text
                x={x + 6}
                y={height / 2 + 4}
                fontSize={11}
                fill="#1e293b"
                className="select-none"
              >
                {displayName}
              </text>
            )}
            {isEditing && w > 40 && (
              <foreignObject x={x + 2} y={2} width={w - 4} height={height - 4}>
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commitEdit}
                  onKeyDown={(e) => e.key === 'Enter' && commitEdit()}
                  style={{
                    width: '100%',
                    height: '100%',
                    fontSize: 11,
                    border: 'none',
                    background: 'transparent',
                    outline: 'none',
                  }}
                />
              </foreignObject>
            )}
          </g>
        );
      })}
    </svg>
  );
}
