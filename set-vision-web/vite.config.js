import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/ws": {
        target: "http://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/settings": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/status": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/control": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
