/**
 * Vitest setup file.
 *
 * Configures jsdom environment and testing-library matchers.
 * Install dependencies before running:
 *   npm install -D vitest @testing-library/react @testing-library/jest-dom @vitejs/plugin-react jsdom
 */

import "@testing-library/jest-dom/vitest";

// Radix UI components (Slider, etc.) use ResizeObserver which is not available in jsdom.
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
