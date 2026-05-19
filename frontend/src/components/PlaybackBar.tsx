import { RefObject, useEffect, useRef, useState } from 'react';

interface Props {
  audioRef: RefObject<HTMLAudioElement | null>;
  durationSec: number;
  timelineWidth: number;
  labelWidth?: number;
}

export function PlaybackBar({ audioRef, durationSec, timelineWidth, labelWidth = 96 }: Props) {
  const [playing, setPlaying] = useState(false);
  const [cursorPx, setCursorPx] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onPause);
    return () => {
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onPause);
    };
  }, [audioRef]);

  useEffect(() => {
    if (!playing) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    const tick = () => {
      const audio = audioRef.current;
      if (audio && durationSec > 0) {
        setCursorPx((audio.currentTime / durationSec) * timelineWidth);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, durationSec, timelineWidth, audioRef]);

  function toggle() {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) audio.play();
    else audio.pause();
  }

  function onScrub(e: React.MouseEvent<HTMLDivElement>) {
    const audio = audioRef.current;
    if (!audio || durationSec <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    audio.currentTime = (x / timelineWidth) * durationSec;
  }

  return (
    <div className="flex items-center mt-2">
      <div style={{ width: labelWidth }} className="pr-2 text-right">
        <button
          type="button"
          aria-label={playing ? 'pause' : 'play'}
          onClick={toggle}
          className="text-xs px-2 py-1 rounded border hover:bg-muted"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
      </div>
      <div
        className="relative bg-muted/30 rounded cursor-pointer"
        style={{ width: timelineWidth, height: 8 }}
        onClick={onScrub}
      >
        <div
          className="absolute top-0 bottom-0 bg-primary"
          style={{ left: 0, width: `${cursorPx}px` }}
        />
      </div>
    </div>
  );
}
