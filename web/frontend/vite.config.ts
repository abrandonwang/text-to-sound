import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the built site works from any GitHub Pages sub-path
// (e.g. https://abrandonwang.github.io/text-to-sound/).
export default defineConfig({
  base: './',
  plugins: [react()],
})
