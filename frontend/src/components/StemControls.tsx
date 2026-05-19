interface Props {
  soloed: boolean;
  muted: boolean;
  onToggleSolo: () => void;
  onToggleMute: () => void;
}

export function StemControls({ soloed, muted, onToggleSolo, onToggleMute }: Props) {
  return (
    <div className="flex gap-1">
      <button
        type="button"
        aria-label="solo"
        aria-pressed={soloed}
        onClick={onToggleSolo}
        className={
          'text-[10px] font-bold w-5 h-5 rounded border ' +
          (soloed ? 'bg-yellow-400 text-black border-yellow-400' : 'border-muted text-muted-foreground hover:bg-muted/40')
        }
      >
        S
      </button>
      <button
        type="button"
        aria-label="mute"
        aria-pressed={muted}
        onClick={onToggleMute}
        className={
          'text-[10px] font-bold w-5 h-5 rounded border ' +
          (muted ? 'bg-red-500 text-white border-red-500' : 'border-muted text-muted-foreground hover:bg-muted/40')
        }
      >
        M
      </button>
    </div>
  );
}
