interface Props {
  durationSec: number;
  bpm: number;
  width: number;
  height: number;
  mode: 'seconds' | 'bars';
}

export function TimeAxis({ durationSec, bpm, width, height, mode }: Props) {
  const ticks: { x: number; label: string }[] = [];

  if (mode === 'seconds') {
    const step = durationSec > 120 ? 30 : durationSec > 60 ? 10 : 5;
    for (let t = 0; t <= durationSec; t += step) {
      ticks.push({ x: (t / durationSec) * width, label: `${t}s` });
    }
  } else {
    const secPerBeat = 60 / (bpm || 120);
    const secPerBar = secPerBeat * 4;
    const totalBars = Math.floor(durationSec / secPerBar);
    const step = totalBars > 64 ? 8 : totalBars > 32 ? 4 : 2;
    for (let bar = 0; bar <= totalBars; bar += step) {
      const t = bar * secPerBar;
      ticks.push({ x: (t / durationSec) * width, label: `${bar + 1}` });
    }
  }

  return (
    <svg width={width} height={height} className="block">
      <line x1={0} y1={0} x2={width} y2={0} stroke="#e2e8f0" strokeWidth={1} />
      {ticks.map(({ x, label }) => (
        <g key={label}>
          <line x1={x} y1={0} x2={x} y2={6} stroke="#94a3b8" strokeWidth={1} />
          <text x={x + 2} y={height - 2} fontSize={9} fill="#94a3b8">
            {label}
          </text>
        </g>
      ))}
    </svg>
  );
}
