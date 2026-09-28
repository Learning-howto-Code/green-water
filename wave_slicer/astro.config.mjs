import { defineConfig } from "astro/config";

// Static site. On Vercel the Python function in api/ serves /api/slice.
// Locally, `npm run api` starts the same function on port 8000 and the dev
// server forwards /api to it.
export default defineConfig({
  devToolbar: { enabled: false },
  vite: {
    server: {
      proxy: { "/api": "http://localhost:8000" },
    },
  },
});
