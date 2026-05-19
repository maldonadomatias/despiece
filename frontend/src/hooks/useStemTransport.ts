import { useCallback, useEffect, useRef, useState } from 'react';

export interface StemTransport {
  registerAudio: (name: string, el: HTMLAudioElement | null) => void;
  toggle: () => void;
  seek: (sec: number) => void;
  playFrom: (sec: number) => void;
  playing: boolean;
  currentTime: number;
}

const DRIFT_THRESHOLD_SEC = 0.08;

export function useStemTransport(): StemTransport {
  const refs = useRef<Map<string, HTMLAudioElement>>(new Map());
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const rafRef = useRef<number | null>(null);

  const registerAudio = useCallback((name: string, el: HTMLAudioElement | null) => {
    if (el) refs.current.set(name, el);
    else refs.current.delete(name);
  }, []);

  const toggle = useCallback(() => {
    const els = Array.from(refs.current.values());
    if (els.length === 0) return;
    const allPaused = els.every((e) => e.paused);
    if (allPaused) {
      setPlaying(true);
      els.forEach((e) => {
        const p = e.play();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      });
    } else {
      setPlaying(false);
      els.forEach((e) => e.pause());
    }
  }, []);

  const seek = useCallback((sec: number) => {
    setCurrentTime(sec);
    refs.current.forEach((e) => {
      e.currentTime = sec;
    });
  }, []);

  const playFrom = useCallback(
    (sec: number) => {
      seek(sec);
      setPlaying(true);
      refs.current.forEach((e) => {
        const p = e.play();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      });
    },
    [seek]
  );

  useEffect(() => {
    if (!playing) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    const tick = () => {
      const els = Array.from(refs.current.values());
      const master = els[0];
      if (master) {
        if (master.ended) {
          setPlaying(false);
          els.forEach((e) => e.pause());
          return;
        }
        setCurrentTime(master.currentTime);
        for (let i = 1; i < els.length; i++) {
          const e = els[i];
          if (Math.abs(e.currentTime - master.currentTime) > DRIFT_THRESHOLD_SEC) {
            e.currentTime = master.currentTime;
          }
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing]);

  return { registerAudio, toggle, seek, playFrom, playing, currentTime };
}
