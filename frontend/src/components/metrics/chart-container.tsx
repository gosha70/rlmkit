"use client";

import { useEffect, useRef, useState } from "react";

interface ChartContainerProps {
  className?: string;
  children: (width: number, height: number) => React.ReactNode;
  "aria-label"?: string;
  role?: string;
}

/**
 * Measures its own layout size and passes (width, height) to children
 * via a render function.  Replaces Recharts' ResponsiveContainer to
 * avoid the "width(-1) and height(-1)" warning that fires when
 * ResponsiveContainer measures its parent before CSS layout is applied.
 *
 * Usage:
 *   <ChartContainer className="h-64">
 *     {(w, h) => <BarChart width={w} height={h} ...> ... </BarChart>}
 *   </ChartContainer>
 */
export function ChartContainer({ className, children, ...rest }: ChartContainerProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const measure = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w > 0 && h > 0) {
        setSize({ width: w, height: h });
      }
    };

    // Immediate check
    measure();

    // Observe for resize (handles initial layout + window resize)
    const observer = new ResizeObserver(() => measure());
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} className={className} {...rest}>
      {size ? children(size.width, size.height) : null}
    </div>
  );
}
