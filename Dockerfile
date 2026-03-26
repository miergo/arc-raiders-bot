# =============================================================================
# Multi-stage Dockerfile — ARC Raiders WikiBot
# =============================================================================
#
# WHY MULTI-STAGE?
#   We need Node.js to build the React frontend and Python to run the backend.
#   Multi-stage builds let us use a Node image for the build step, then copy
#   only the built static files into the final Python image. This keeps the
#   final image small (no Node.js, no node_modules in production).
#
# HOW IT WORKS:
#   Stage 1 (frontend-build): Installs npm deps and runs `npm run build`,
#       producing static HTML/JS/CSS in frontend/dist/.
#   Stage 2 (runtime): Installs Python deps, copies backend code + the built
#       frontend, and starts the FastAPI server on port 8000.
#
# The server.py is configured to serve the frontend's static files when
# frontend/dist/ exists, so this single container serves both the API and UI.
#
# USAGE:
#   docker compose up              (recommended — see docker-compose.yml)
#   docker build -t arc-bot .      (standalone build)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build the React frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build

WORKDIR /build/frontend

# Copy package files first for better Docker layer caching.
# Docker caches each layer — if package.json hasn't changed, npm ci is skipped
# on rebuild, saving minutes of install time.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Now copy the rest of the frontend source and build it.
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python runtime (final image)
# ---------------------------------------------------------------------------
FROM python:3.13-slim

# Set working directory to /app — all paths below are relative to this.
WORKDIR /app

# Install Python dependencies first (layer caching — same idea as above).
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code.
COPY backend/ ./

# Copy the built frontend from Stage 1 into a location where server.py
# can find and serve it.
COPY --from=frontend-build /build/frontend/dist ./frontend_dist

# Tell the server where to find the frontend build output.
ENV FRONTEND_DIST_DIR=/app/frontend_dist

# Expose port 8000 — this is the FastAPI server port.
EXPOSE 8000

# Start the server. Using uvicorn directly (not `python server.py`) gives
# us more control over the process and is the standard production pattern.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
