import { useEffect, useEffectEvent } from "react";

export function useInterval(callback: () => void, delay: number | null): void {
  const onTick = useEffectEvent(callback);

  useEffect(() => {
    if (delay === null) {
      return;
    }

    const id = window.setInterval(() => {
      onTick();
    }, delay);

    return () => window.clearInterval(id);
  }, [delay]);
}
