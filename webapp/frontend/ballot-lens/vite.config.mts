import { fileURLToPath, URL } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const appRoot = fileURLToPath(new URL('./', import.meta.url));
const outDir = fileURLToPath(
  new URL('../../static/dist/ballot-lens-f2/', import.meta.url),
);

export default defineConfig({
  root: appRoot,
  base: '/static/dist/ballot-lens-f2/',
  plugins: [react()],
  build: {
    target: 'baseline-widely-available',
    outDir,
    emptyOutDir: true,
    manifest: 'manifest.json',
    sourcemap: false,
    cssCodeSplit: false,
    rollupOptions: {
      input: fileURLToPath(new URL('./main.tsx', import.meta.url)),
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
});
