# Song Structure Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full-stack app that accepts an audio file, separates stems with Demucs, analyzes BPM/key/structure, stores results in Postgres, and renders an interactive SVG timeline showing when each stem is active.

**Architecture:** Express backend handles upload (→ MinIO) + job queue (Postgres `FOR UPDATE SKIP LOCKED`); Python FastAPI `audio-service` does the heavy compute (Demucs 4 + librosa + MSAF), returning JSON; React frontend polls for job completion and renders an SVG Gantt-style timeline.

**Tech Stack:** React 19 / Vite / TypeScript, Node 20 / Express 4 / TypeScript, Python 3.11 / FastAPI / Demucs 4 / librosa 0.10 / MSAF, PostgreSQL 15, MinIO (S3-compat), `@aws-sdk/client-s3` (Node), `boto3` (Python), `multer` (upload), SVG (timeline)

---

## File Map

### New files

```
audio-service/
  Dockerfile
  requirements.txt
  src/
    main.py                  # FastAPI app, /health, /analyze
    models.py                # Pydantic request/response models
    storage.py               # boto3 S3/MinIO client
    analysis/
      demucs_runner.py       # Demucs stem separation
      envelope.py            # RMS envelope extraction
      beat_analysis.py       # BPM, beat grid, key detection
      segmentation.py        # Structural segmentation (MSAF + fallback)
  tests/
    conftest.py
    test_envelope.py
    test_beat_analysis.py
    test_main.py

backend/src/
  domain/song.ts             # Song + job TypeScript types
  services/
    storageService.ts        # @aws-sdk/client-s3 wrapper (upload/delete/presign)
    songService.ts           # CRUD: songs + song_analysis rows
    audioClient.ts           # fetch wrapper to call audio-service /analyze
  routes/songs.ts            # Express router: POST/GET/DELETE /api/songs
  workers/jobWorker.ts       # setInterval polling + FOR UPDATE SKIP LOCKED
  db/migrations/
    002_songs_schema.sql

frontend/src/
  types/song.ts              # Shared Song / AnalysisResult types
  lib/api.ts                 # Axios wrappers for songs API
  pages/
    HomePage.tsx             # Upload drop zone + songs list
    AnalysisPage.tsx         # Timeline view with polling
  components/
    AudioUpload.tsx          # Drag-and-drop file input
    SongTimeline.tsx         # Outer container (SVG layout)
    StemRow.tsx              # Single stem energy track (SVG)
    SectionBar.tsx           # Section labels across top (SVG)
    TimeAxis.tsx             # Compases / seconds axis toggle
    SongCard.tsx             # List card (name, BPM, key, status)
```

### Modified files

```
docker-compose.yml           # +minio +audio-service services
docker-compose.dev.yml       # dev overrides for new services
backend/env.example          # +S3, +AUDIO_SERVICE_URL vars
backend/src/app.ts           # register /api/songs router
backend/src/index.ts         # start jobWorker after server ready
frontend/src/App.tsx         # add /songs/:id route
```

---

## Phase 0 — Infrastructure

### Task 1: Extend Docker Compose (MinIO + audio-service stub)

**Files:**
- Modify: `docker-compose.yml`
- Modify: `docker-compose.dev.yml`
- Modify: `backend/env.example`

- [ ] **Step 1: Add MinIO and audio-service to `docker-compose.yml`**

Replace the file with:

```yaml
services:
  frontend:
    build:
      context: ./frontend
      args:
        - VITE_API_URL=/api
    ports:
      - '3000:3000'
    depends_on:
      backend:
        condition: service_healthy
    environment:
      - VITE_API_URL=/api
      - BACKEND_URL=http://backend:5000
      - BACKEND_DOMAIN=localhost

  backend:
    build: ./backend
    ports:
      - '5001:5000'
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
    env_file:
      - ./backend/.env
    environment:
      - PORT=5000
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL:-postgres://user:password@postgres:5432/mydb}
      - FRONTEND_URL=http://localhost:3000
      - LOG_LEVEL=info
      - S3_ENDPOINT=http://minio:9000
      - S3_ACCESS_KEY=minioadmin
      - S3_SECRET_KEY=minioadmin
      - S3_BUCKET=songs
      - S3_REGION=us-east-1
      - AUDIO_SERVICE_URL=http://audio-service:8000
    healthcheck:
      test: ['CMD', 'node', '-e', "require('http').get('http://localhost:5000/api/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  audio-service:
    build: ./audio-service
    ports:
      - '8000:8000'
    environment:
      - S3_ENDPOINT=http://minio:9000
      - S3_ACCESS_KEY=minioadmin
      - S3_SECRET_KEY=minioadmin
      - S3_BUCKET=songs
      - S3_REGION=us-east-1
    depends_on:
      minio:
        condition: service_healthy
    healthcheck:
      test: ['CMD', 'curl', '-f', 'http://localhost:8000/health']
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 60s

  minio:
    image: minio/minio:latest
    command: server /data --console-address ':9001'
    ports:
      - '9000:9000'
      - '9001:9001'
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio-data:/data
    healthcheck:
      test: ['CMD', 'curl', '-f', 'http://localhost:9000/minio/health/live']
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
    ports:
      - '5433:5432'
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ['CMD-SHELL', 'pg_isready -U user -d mydb']
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s

volumes:
  postgres-data:
  minio-data:
```

- [ ] **Step 2: Update `docker-compose.dev.yml` for dev overrides**

```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    volumes:
      - ./frontend:/app
      - /app/node_modules
    ports:
      - '3000:3000'
    environment:
      - VITE_API_URL=http://localhost:5001/api

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    volumes:
      - ./backend:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgres://user:password@postgres:5432/mydb
      - S3_ENDPOINT=http://minio:9000
      - S3_ACCESS_KEY=minioadmin
      - S3_SECRET_KEY=minioadmin
      - S3_BUCKET=songs
      - S3_REGION=us-east-1
      - AUDIO_SERVICE_URL=http://audio-service:8000
    ports:
      - '5001:5000'

  audio-service:
    build:
      context: ./audio-service
      dockerfile: Dockerfile
    volumes:
      - demucs-models:/root/.cache/torch
    environment:
      - S3_ENDPOINT=http://minio:9000
      - S3_ACCESS_KEY=minioadmin
      - S3_SECRET_KEY=minioadmin
      - S3_BUCKET=songs
      - S3_REGION=us-east-1

  postgres:
    ports:
      - '5433:5432'

  minio:
    ports:
      - '9000:9000'
      - '9001:9001'

volumes:
  demucs-models:
```

- [ ] **Step 3: Update `backend/env.example`**

```bash
PORT=5001
NODE_ENV=development
DATABASE_URL=postgres://user:password@localhost:5433/mydb
FRONTEND_URL=http://localhost:3000
LOG_LEVEL=info
# Object storage (MinIO locally, S3/R2/Spaces in prod)
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=songs
S3_REGION=us-east-1
# Audio analysis service
AUDIO_SERVICE_URL=http://localhost:8000
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml docker-compose.dev.yml backend/env.example
git commit -m "chore: add MinIO and audio-service to docker-compose"
```

---

### Task 2: Audio Service — FastAPI skeleton

**Files:**
- Create: `audio-service/requirements.txt`
- Create: `audio-service/Dockerfile`
- Create: `audio-service/src/main.py`
- Create: `audio-service/src/models.py`

- [ ] **Step 1: Create `audio-service/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
boto3==1.35.0
demucs==4.0.1
librosa==0.10.2
msaf==0.1.75
soundfile==0.12.1
numpy==1.26.4
torch==2.4.0
torchaudio==2.4.0
pydantic==2.9.0
python-multipart==0.0.12
httpx==0.27.2
pytest==8.3.3
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Create `audio-service/Dockerfile`**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Demucs htdemucs model (~200MB) so it's cached in the image layer
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')" || true

COPY src/ ./src/

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `audio-service/src/models.py`**

```python
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    storage_key: str


class EnvelopePoint(BaseModel):
    t: float
    e: float


class Section(BaseModel):
    label: str
    start_sec: float
    end_sec: float


class StemAnalysis(BaseModel):
    envelope: list[list[float]]  # [[time_sec, energy_0_1], ...]


class AnalysisResult(BaseModel):
    bpm: float
    key: str
    duration_sec: float
    beat_grid: list[float]
    sections: list[dict]
    stems: dict[str, dict]
```

- [ ] **Step 4: Create `audio-service/src/main.py`**

```python
from fastapi import FastAPI, HTTPException
from .models import AnalyzeRequest, AnalysisResult

app = FastAPI(title="Audio Analysis Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(req: AnalyzeRequest):
    # Implemented in Phase 1 Task 6
    raise HTTPException(status_code=501, detail="Not yet implemented")
```

- [ ] **Step 5: Create `audio-service/tests/conftest.py`**

```python
import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """440 Hz sine wave, 5 seconds, sr=22050."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y, sr
```

- [ ] **Step 6: Verify audio service builds**

```bash
cd audio-service && docker build -t audio-service-test . 2>&1 | tail -5
```

Expected: `Successfully built ...` (may take several minutes due to Demucs model download)

- [ ] **Step 7: Commit**

```bash
git add audio-service/
git commit -m "feat(audio-service): FastAPI skeleton with /health endpoint"
```

---

### Task 3: Backend — Storage Service (MinIO/S3)

**Files:**
- Create: `backend/src/services/storageService.ts`

- [ ] **Step 1: Install AWS SDK in backend**

```bash
cd backend && npm install @aws-sdk/client-s3 multer @types/multer file-type
```

- [ ] **Step 2: Create `backend/src/services/storageService.ts`**

```typescript
import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  GetObjectCommand,
} from '@aws-sdk/client-s3';
import logger from '../utils/logger.js';

function buildS3Client(): S3Client {
  return new S3Client({
    endpoint: process.env.S3_ENDPOINT!,
    region: process.env.S3_REGION ?? 'us-east-1',
    credentials: {
      accessKeyId: process.env.S3_ACCESS_KEY!,
      secretAccessKey: process.env.S3_SECRET_KEY!,
    },
    forcePathStyle: true, // required for MinIO
  });
}

const s3 = buildS3Client();
const BUCKET = process.env.S3_BUCKET ?? 'songs';

export async function ensureBucketExists(): Promise<void> {
  const { CreateBucketCommand, HeadBucketCommand } = await import(
    '@aws-sdk/client-s3'
  );
  try {
    await s3.send(new HeadBucketCommand({ Bucket: BUCKET }));
  } catch {
    await s3.send(new CreateBucketCommand({ Bucket: BUCKET }));
    logger.info({ bucket: BUCKET }, 'Created S3 bucket');
  }
}

export async function uploadFile(
  key: string,
  buffer: Buffer,
  contentType: string
): Promise<void> {
  await s3.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: key,
      Body: buffer,
      ContentType: contentType,
    })
  );
}

export async function deleteFile(key: string): Promise<void> {
  await s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: key }));
}
```

- [ ] **Step 3: Write test for storageService (mocked)**

Create `backend/src/services/__tests__/storageService.test.ts`:

```typescript
jest.mock('@aws-sdk/client-s3', () => ({
  S3Client: jest.fn().mockImplementation(() => ({
    send: jest.fn().mockResolvedValue({}),
  })),
  PutObjectCommand: jest.fn(),
  DeleteObjectCommand: jest.fn(),
  GetObjectCommand: jest.fn(),
  HeadBucketCommand: jest.fn(),
  CreateBucketCommand: jest.fn(),
}));

import { uploadFile, deleteFile } from '../storageService.js';

describe('storageService', () => {
  it('uploadFile resolves without throwing', async () => {
    await expect(
      uploadFile('test/key.mp3', Buffer.from('data'), 'audio/mpeg')
    ).resolves.toBeUndefined();
  });

  it('deleteFile resolves without throwing', async () => {
    await expect(deleteFile('test/key.mp3')).resolves.toBeUndefined();
  });
});
```

- [ ] **Step 4: Run test**

```bash
cd backend && npm test -- --testPathPattern=storageService
```

Expected: `PASS src/services/__tests__/storageService.test.ts`

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/storageService.ts backend/src/services/__tests__/storageService.test.ts backend/package.json backend/package-lock.json
git commit -m "feat(backend): add S3/MinIO storage service"
```

---

### Task 4: Backend — Database migration 002

**Files:**
- Create: `backend/src/db/migrations/002_songs_schema.sql`

- [ ] **Step 1: Create migration**

```sql
-- Songs: one row per uploaded audio file
CREATE TABLE IF NOT EXISTS songs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  storage_key TEXT NOT NULL,
  original_name TEXT NOT NULL,
  duration_sec NUMERIC(10,3),
  bpm NUMERIC(6,2),
  music_key TEXT,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued','processing','done','error')),
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Job queue: one row per analysis job
CREATE TABLE IF NOT EXISTS song_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued','processing','done','error')),
  attempts INT NOT NULL DEFAULT 0,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_song_jobs_status ON song_jobs(status, created_at)
  WHERE status = 'queued';

-- Analysis results: JSONB payload returned by audio-service
CREATE TABLE IF NOT EXISTS song_analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
  result_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_song_analysis_song_id ON song_analysis(song_id);
```

- [ ] **Step 2: Copy env and run migration locally (requires Postgres running)**

```bash
cd backend && cp env.example .env
# Edit .env so DATABASE_URL matches your local postgres (port 5433)
npm run db:migrate
```

Expected: `Executed migration: 002_songs_schema.sql`

- [ ] **Step 3: Commit**

```bash
git add backend/src/db/migrations/002_songs_schema.sql
git commit -m "feat(backend): add songs/song_jobs/song_analysis schema"
```

---

## Phase 1 — Audio Pipeline

### Task 5: Backend — Song upload endpoint

**Files:**
- Create: `backend/src/domain/song.ts`
- Create: `backend/src/services/songService.ts`
- Create: `backend/src/routes/songs.ts`
- Modify: `backend/src/app.ts`

- [ ] **Step 1: Create `backend/src/domain/song.ts`**

```typescript
export type SongStatus = 'queued' | 'processing' | 'done' | 'error';

export interface Song {
  id: string;
  storage_key: string;
  original_name: string;
  duration_sec: number | null;
  bpm: number | null;
  music_key: string | null;
  status: SongStatus;
  error_message: string | null;
  created_at: string;
}

export interface SongAnalysis {
  id: string;
  song_id: string;
  result_json: AnalysisResult;
  created_at: string;
}

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  sections: Section[];
  stems: Record<string, StemData>;
}

export interface Section {
  label: string;
  start_sec: number;
  end_sec: number;
}

export interface StemData {
  envelope: [number, number][]; // [time_sec, energy_0_1]
}
```

- [ ] **Step 2: Create `backend/src/services/songService.ts`**

```typescript
import pool from '../db/connect.js';
import { Song, SongAnalysis, AnalysisResult } from '../domain/song.js';

export async function createSong(
  storageKey: string,
  originalName: string
): Promise<Song> {
  const { rows } = await pool.query<Song>(
    `INSERT INTO songs (storage_key, original_name)
     VALUES ($1, $2) RETURNING *`,
    [storageKey, originalName]
  );
  return rows[0];
}

export async function getSong(id: string): Promise<Song | null> {
  const { rows } = await pool.query<Song>(
    'SELECT * FROM songs WHERE id = $1',
    [id]
  );
  return rows[0] ?? null;
}

export async function listSongs(): Promise<Song[]> {
  const { rows } = await pool.query<Song>(
    'SELECT * FROM songs ORDER BY created_at DESC'
  );
  return rows;
}

export async function getSongAnalysis(
  songId: string
): Promise<SongAnalysis | null> {
  const { rows } = await pool.query<SongAnalysis>(
    'SELECT * FROM song_analysis WHERE song_id = $1',
    [songId]
  );
  return rows[0] ?? null;
}

export async function createJob(songId: string): Promise<string> {
  const { rows } = await pool.query(
    `INSERT INTO song_jobs (song_id) VALUES ($1) RETURNING id`,
    [songId]
  );
  return rows[0].id;
}

export async function deleteSong(id: string): Promise<string | null> {
  const { rows } = await pool.query<{ storage_key: string }>(
    'DELETE FROM songs WHERE id = $1 RETURNING storage_key',
    [id]
  );
  return rows[0]?.storage_key ?? null;
}
```

- [ ] **Step 3: Create `backend/src/routes/songs.ts`**

```typescript
import { Router, Request, Response, NextFunction } from 'express';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';
import { fileTypeFromBuffer } from 'file-type';
import { uploadFile, deleteFile } from '../services/storageService.js';
import * as songService from '../services/songService.js';
import logger from '../utils/logger.js';

const router = Router();

const ALLOWED_MIMES = new Set([
  'audio/mpeg',
  'audio/wav',
  'audio/x-wav',
  'audio/flac',
  'audio/x-flac',
]);

const MAX_BYTES = 200 * 1024 * 1024; // 200 MB

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_BYTES },
  fileFilter: (_req, file, cb) => {
    const ext = file.originalname.split('.').pop()?.toLowerCase();
    if (!['mp3', 'wav', 'flac'].includes(ext ?? '')) {
      return cb(new Error('Only mp3, wav, and flac files are allowed'));
    }
    cb(null, true);
  },
});

// POST /api/songs
router.post(
  '/',
  upload.single('file'),
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'No file uploaded' });
        return;
      }

      // Validate actual file content (magic bytes)
      const type = await fileTypeFromBuffer(req.file.buffer);
      if (!type || !ALLOWED_MIMES.has(type.mime)) {
        res.status(400).json({ error: 'Invalid audio file content' });
        return;
      }

      const storageKey = `songs/${uuidv4()}.${type.ext}`;
      await uploadFile(storageKey, req.file.buffer, type.mime);

      const song = await songService.createSong(storageKey, req.file.originalname);
      await songService.createJob(song.id);

      logger.info({ songId: song.id }, 'Song upload queued');
      res.status(201).json({ id: song.id, status: song.status });
    } catch (err) {
      next(err);
    }
  }
);

// GET /api/songs
router.get('/', async (_req: Request, res: Response, next: NextFunction) => {
  try {
    const songs = await songService.listSongs();
    res.json(songs);
  } catch (err) {
    next(err);
  }
});

// GET /api/songs/:id
router.get('/:id', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const song = await songService.getSong(req.params.id);
    if (!song) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    const result: Record<string, unknown> = { ...song };
    if (song.status === 'done') {
      const analysis = await songService.getSongAnalysis(song.id);
      result.analysis = analysis?.result_json ?? null;
    }
    res.json(result);
  } catch (err) {
    next(err);
  }
});

// DELETE /api/songs/:id
router.delete('/:id', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const storageKey = await songService.deleteSong(req.params.id);
    if (!storageKey) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    await deleteFile(storageKey).catch((err) =>
      logger.warn({ err, storageKey }, 'Failed to delete from storage')
    );
    res.status(204).send();
  } catch (err) {
    next(err);
  }
});

export default router;
```

- [ ] **Step 4: Register routes in `backend/src/app.ts`**

Add after the health check route and before the error handler:

```typescript
import songsRouter from './routes/songs.js';
// ...
app.use('/api/songs', songsRouter);
```

- [ ] **Step 5: Write route tests**

Create `backend/src/routes/__tests__/songs.test.ts`:

```typescript
import request from 'supertest';
import app from '../../app.js';

jest.mock('../../services/storageService.js', () => ({
  uploadFile: jest.fn().mockResolvedValue(undefined),
  deleteFile: jest.fn().mockResolvedValue(undefined),
  ensureBucketExists: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('../../services/songService.js', () => ({
  createSong: jest.fn().mockResolvedValue({
    id: 'test-id',
    status: 'queued',
    storage_key: 'songs/test.mp3',
    original_name: 'test.mp3',
    created_at: new Date().toISOString(),
    bpm: null,
    music_key: null,
    duration_sec: null,
    error_message: null,
  }),
  createJob: jest.fn().mockResolvedValue('job-id'),
  listSongs: jest.fn().mockResolvedValue([]),
  getSong: jest.fn().mockResolvedValue(null),
  deleteSong: jest.fn().mockResolvedValue(null),
  getSongAnalysis: jest.fn().mockResolvedValue(null),
}));

describe('GET /api/songs', () => {
  it('returns empty array', async () => {
    const res = await request(app).get('/api/songs');
    expect(res.status).toBe(200);
    expect(res.body).toEqual([]);
  });
});

describe('GET /api/songs/:id', () => {
  it('returns 404 for unknown id', async () => {
    const res = await request(app).get('/api/songs/unknown-id');
    expect(res.status).toBe(404);
  });
});
```

Install `supertest`:
```bash
cd backend && npm install --save-dev supertest @types/supertest
```

- [ ] **Step 6: Run tests**

```bash
cd backend && npm test -- --testPathPattern=songs
```

Expected: `PASS src/routes/__tests__/songs.test.ts`

- [ ] **Step 7: Commit**

```bash
git add backend/src/domain/song.ts backend/src/services/songService.ts backend/src/routes/songs.ts backend/src/app.ts backend/package.json backend/package-lock.json backend/src/routes/__tests__/songs.test.ts
git commit -m "feat(backend): song upload endpoint + CRUD services"
```

---

### Task 6: Audio Service — Demucs + RMS envelope

**Files:**
- Create: `audio-service/src/storage.py`
- Create: `audio-service/src/analysis/demucs_runner.py`
- Create: `audio-service/src/analysis/envelope.py`
- Create: `audio-service/tests/test_envelope.py`

- [ ] **Step 1: Create `audio-service/src/storage.py`**

```python
import os
import boto3
from botocore.client import Config


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )


def download_to_path(storage_key: str, dest_path: str) -> None:
    client = get_s3_client()
    bucket = os.environ["S3_BUCKET"]
    client.download_file(bucket, storage_key, dest_path)
```

- [ ] **Step 2: Create `audio-service/src/analysis/demucs_runner.py`**

```python
import os
import numpy as np
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile

# Stem order in htdemucs
STEMS = ["drums", "bass", "other", "vocals"]

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = get_model("htdemucs")
        _model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
    return _model


def separate_stems(
    input_path: str, out_dir: str
) -> dict[str, tuple[np.ndarray, int]]:
    """Run Demucs on input_path; return {stem_name: (mono_float32, sample_rate)}."""
    model = _get_model()
    device = next(model.parameters()).device

    wav = AudioFile(input_path).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    wav = wav.unsqueeze(0).to(device)  # (1, channels, samples)

    with torch.no_grad():
        sources = apply_model(model, wav, device=device)[0]  # (4, channels, samples)

    results: dict[str, tuple[np.ndarray, int]] = {}
    for i, stem_name in enumerate(STEMS):
        stem_wav = sources[i]  # (channels, samples)
        mono = stem_wav.mean(dim=0).cpu().numpy()  # (samples,) float32
        results[stem_name] = (mono, model.samplerate)

    return results
```

- [ ] **Step 3: Create `audio-service/src/analysis/envelope.py`**

```python
import numpy as np
import librosa


def compute_envelope(
    audio: np.ndarray, sr: int, target_fps: float = 25.0
) -> list[list[float]]:
    """Return [[time_sec, energy_0_1], ...] at target_fps frames per second."""
    hop_length = max(1, int(sr / target_fps))
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    max_rms = float(rms.max())
    if max_rms > 0:
        rms = rms / max_rms

    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    return [[round(float(t), 4), round(float(e), 4)] for t, e in zip(times, rms)]
```

- [ ] **Step 4: Write envelope tests**

Create `audio-service/tests/test_envelope.py`:

```python
import numpy as np
import pytest
from src.analysis.envelope import compute_envelope


def test_returns_list_of_pairs(sample_audio):
    y, sr = sample_audio
    result = compute_envelope(y, sr, target_fps=25.0)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(len(p) == 2 for p in result)


def test_energy_normalized_0_to_1(sample_audio):
    y, sr = sample_audio
    result = compute_envelope(y, sr, target_fps=25.0)
    energies = [p[1] for p in result]
    assert max(energies) <= 1.0
    assert min(energies) >= 0.0


def test_silence_has_zero_energy():
    sr = 22050
    silence = np.zeros(sr * 3, dtype=np.float32)
    result = compute_envelope(silence, sr)
    assert all(p[1] == 0.0 for p in result)


def test_time_increases_monotonically(sample_audio):
    y, sr = sample_audio
    result = compute_envelope(y, sr)
    times = [p[0] for p in result]
    assert times == sorted(times)
```

- [ ] **Step 5: Run tests**

```bash
cd audio-service && pip install -r requirements.txt -q && python -m pytest tests/test_envelope.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add audio-service/src/storage.py audio-service/src/analysis/ audio-service/tests/test_envelope.py
git commit -m "feat(audio-service): Demucs stem separation + RMS envelope"
```

---

### Task 7: Audio Service — /analyze endpoint (Phase 1: stems only)

**Files:**
- Modify: `audio-service/src/main.py`
- Create: `audio-service/tests/test_main.py`

- [ ] **Step 1: Update `audio-service/src/main.py`**

```python
import os
import tempfile
import traceback

from fastapi import FastAPI, HTTPException
from .models import AnalyzeRequest, AnalysisResult
from .storage import download_to_path
from .analysis.demucs_runner import separate_stems
from .analysis.envelope import compute_envelope

app = FastAPI(title="Audio Analysis Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.audio")

        try:
            download_to_path(req.storage_key, input_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Cannot fetch audio: {e}")

        try:
            stems_data = separate_stems(input_path, tmpdir)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Demucs failed: {e}")

        stems: dict = {}
        duration_sec = 0.0
        for stem_name, (audio, sr) in stems_data.items():
            if duration_sec == 0.0:
                duration_sec = len(audio) / sr
            stems[stem_name] = {"envelope": compute_envelope(audio, sr)}

        return {
            "bpm": 0.0,          # populated in Phase 2
            "key": "unknown",    # populated in Phase 2
            "duration_sec": round(duration_sec, 3),
            "beat_grid": [],     # populated in Phase 2
            "sections": [],      # populated in Phase 2
            "stems": stems,
        }
```

- [ ] **Step 2: Create `audio-service/tests/test_main.py`**

```python
import numpy as np
import pytest
import soundfile as sf
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    from src.main import app
    return TestClient(app)


@pytest.fixture
def dummy_wav(tmp_path, sample_audio):
    y, sr = sample_audio
    path = str(tmp_path / "test.wav")
    sf.write(path, y, sr)
    return path


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_analyze_returns_stems(client, dummy_wav, sample_audio):
    y, sr = sample_audio

    def fake_download(key, dest):
        import shutil
        shutil.copy(dummy_wav, dest)

    def fake_separate(path, out_dir):
        return {
            "drums": (y, sr),
            "bass": (y * 0.5, sr),
            "other": (y * 0.3, sr),
            "vocals": (y * 0.2, sr),
        }

    with patch("src.main.download_to_path", side_effect=fake_download), \
         patch("src.main.separate_stems", side_effect=fake_separate):
        res = client.post("/analyze", json={"storage_key": "songs/test.wav"})

    assert res.status_code == 200
    body = res.json()
    assert set(body["stems"].keys()) == {"drums", "bass", "other", "vocals"}
    assert body["duration_sec"] > 0
    assert all("envelope" in v for v in body["stems"].values())
```

- [ ] **Step 3: Run tests**

```bash
cd audio-service && python -m pytest tests/test_main.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add audio-service/src/main.py audio-service/tests/test_main.py
git commit -m "feat(audio-service): /analyze endpoint (stems + envelopes)"
```

---

### Task 8: Backend — Job Worker

**Files:**
- Create: `backend/src/services/audioClient.ts`
- Create: `backend/src/workers/jobWorker.ts`
- Modify: `backend/src/index.ts`

- [ ] **Step 1: Create `backend/src/services/audioClient.ts`**

```typescript
import { AnalysisResult } from '../domain/song.js';
import logger from '../utils/logger.js';

const AUDIO_SERVICE_URL =
  process.env.AUDIO_SERVICE_URL ?? 'http://localhost:8000';

export async function analyzeAudio(storageKey: string): Promise<AnalysisResult> {
  const res = await fetch(`${AUDIO_SERVICE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storage_key: storageKey }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'unknown error');
    throw new Error(`Audio service error ${res.status}: ${text}`);
  }

  return res.json() as Promise<AnalysisResult>;
}
```

- [ ] **Step 2: Create `backend/src/workers/jobWorker.ts`**

```typescript
import pool from '../db/connect.js';
import { analyzeAudio } from '../services/audioClient.js';
import logger from '../utils/logger.js';

const POLL_INTERVAL_MS = 3000;
const MAX_ATTEMPTS = 3;

async function processNextJob(): Promise<void> {
  const client = await pool.connect();
  let jobId: string | undefined;
  let songId: string | undefined;
  let storageKey: string | undefined;

  try {
    await client.query('BEGIN');
    const { rows } = await client.query<{
      id: string;
      song_id: string;
      storage_key: string;
      attempts: number;
    }>(`
      SELECT j.id, j.song_id, j.attempts, s.storage_key
      FROM song_jobs j
      JOIN songs s ON s.id = j.song_id
      WHERE j.status = 'queued' AND j.attempts < $1
      ORDER BY j.created_at
      LIMIT 1
      FOR UPDATE OF j SKIP LOCKED
    `, [MAX_ATTEMPTS]);

    if (rows.length === 0) {
      await client.query('COMMIT');
      return;
    }

    ({ id: jobId, song_id: songId, storage_key: storageKey } = rows[0]);

    await client.query(
      `UPDATE song_jobs SET status='processing', attempts=attempts+1, started_at=NOW() WHERE id=$1`,
      [jobId]
    );
    await client.query(`UPDATE songs SET status='processing' WHERE id=$1`, [songId]);
    await client.query('COMMIT');
  } catch (err) {
    await client.query('ROLLBACK').catch(() => {});
    client.release();
    logger.error({ err }, 'Job worker DB error during acquire');
    return;
  }

  client.release();

  // Process outside transaction so the row is visible to other workers as 'processing'
  try {
    logger.info({ jobId, songId }, 'Processing audio job');
    const result = await analyzeAudio(storageKey!);

    await pool.query(
      `INSERT INTO song_analysis (song_id, result_json) VALUES ($1, $2)
       ON CONFLICT DO NOTHING`,
      [songId, JSON.stringify(result)]
    );
    await pool.query(
      `UPDATE songs SET status='done', bpm=$1, music_key=$2, duration_sec=$3 WHERE id=$4`,
      [result.bpm || null, result.key !== 'unknown' ? result.key : null, result.duration_sec, songId]
    );
    await pool.query(
      `UPDATE song_jobs SET status='done', finished_at=NOW() WHERE id=$1`,
      [jobId]
    );
    logger.info({ jobId, songId }, 'Audio job completed');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    logger.error({ err, jobId, songId }, 'Audio job failed');
    await pool.query(
      `UPDATE song_jobs SET status='queued', error=$1, started_at=NULL WHERE id=$2`,
      [message, jobId]
    ).catch(() => {});
    // Mark as error if max attempts reached
    await pool.query(
      `UPDATE song_jobs SET status='error'
       WHERE id=$1 AND attempts >= $2`,
      [jobId, MAX_ATTEMPTS]
    ).catch(() => {});
    await pool.query(
      `UPDATE songs SET status='error', error_message=$1
       WHERE id=$2 AND NOT EXISTS (
         SELECT 1 FROM song_jobs WHERE song_id=$2 AND status='queued'
       )`,
      [message, songId]
    ).catch(() => {});
  }
}

export function startJobWorker(): void {
  logger.info('Job worker started');
  setInterval(() => {
    processNextJob().catch((err) =>
      logger.error({ err }, 'Unexpected error in job worker')
    );
  }, POLL_INTERVAL_MS);
}
```

- [ ] **Step 3: Modify `backend/src/index.ts` to start the worker**

Read the current `backend/src/index.ts` and add the worker start after the server is listening:

```typescript
import app from './app.js';
import logger from './utils/logger.js';
import { ensureBucketExists } from './services/storageService.js';
import { startJobWorker } from './workers/jobWorker.js';

const PORT = process.env.PORT ?? 5000;

async function main() {
  await ensureBucketExists();
  app.listen(PORT, () => {
    logger.info({ port: PORT }, 'Server started');
    startJobWorker();
  });
}

main().catch((err) => {
  logger.error({ err }, 'Failed to start server');
  process.exit(1);
});
```

- [ ] **Step 4: Build backend to check TS errors**

```bash
cd backend && npm run build 2>&1
```

Expected: exit 0, no errors.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/audioClient.ts backend/src/workers/jobWorker.ts backend/src/index.ts
git commit -m "feat(backend): job worker with FOR UPDATE SKIP LOCKED"
```

---

### Task 9: Frontend — Types + API client

**Files:**
- Create: `frontend/src/types/song.ts`
- Create/Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Create `frontend/src/types/song.ts`**

```typescript
export type SongStatus = 'queued' | 'processing' | 'done' | 'error';

export interface Song {
  id: string;
  original_name: string;
  duration_sec: number | null;
  bpm: number | null;
  music_key: string | null;
  status: SongStatus;
  error_message: string | null;
  created_at: string;
  analysis?: AnalysisResult;
}

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  sections: Section[];
  stems: Record<string, StemData>;
}

export interface Section {
  label: string;
  start_sec: number;
  end_sec: number;
}

export interface StemData {
  envelope: [number, number][]; // [time_sec, energy_0_1]
}
```

- [ ] **Step 2: Create `frontend/src/lib/api.ts`**

```typescript
import axios from 'axios';
import { Song } from '../types/song';

const apiBase = import.meta.env.VITE_API_URL ?? '/api';

const http = axios.create({ baseURL: apiBase });

export async function uploadSong(file: File): Promise<{ id: string; status: string }> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await http.post('/songs', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function getSong(id: string): Promise<Song> {
  const { data } = await http.get<Song>(`/songs/${id}`);
  return data;
}

export async function listSongs(): Promise<Song[]> {
  const { data } = await http.get<Song[]>('/songs');
  return data;
}

export async function deleteSong(id: string): Promise<void> {
  await http.delete(`/songs/${id}`);
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/song.ts frontend/src/lib/api.ts
git commit -m "feat(frontend): song types and API client"
```

---

### Task 10: Frontend — HomePage with upload + polling

**Files:**
- Create: `frontend/src/components/AudioUpload.tsx`
- Create: `frontend/src/pages/HomePage.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create `frontend/src/components/AudioUpload.tsx`**

```tsx
import { useRef, useState, DragEvent, ChangeEvent } from 'react';
import { uploadSong } from '@/lib/api';

const MAX_MB = 200;
const ALLOWED_EXT = ['.mp3', '.wav', '.flac'];

interface Props {
  onUploaded: (id: string) => void;
}

export function AudioUpload({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function validate(file: File): string | null {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) return 'Only mp3, wav, and flac allowed';
    if (file.size > MAX_MB * 1024 * 1024) return `File must be under ${MAX_MB} MB`;
    return null;
  }

  async function handleFile(file: File) {
    const err = validate(file);
    if (err) { setError(err); return; }
    setError(null);
    setUploading(true);
    try {
      const { id } = await uploadSong(file);
      onUploaded(id);
    } catch {
      setError('Upload failed. Try again.');
    } finally {
      setUploading(false);
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        flex flex-col items-center justify-center gap-3 p-12 rounded-xl border-2 border-dashed cursor-pointer transition-colors
        ${dragging ? 'border-primary bg-primary/5' : 'border-muted-foreground/30 hover:border-primary/50'}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac"
        className="hidden"
        onChange={onChange}
      />
      <div className="text-4xl">🎵</div>
      <p className="text-sm text-muted-foreground text-center">
        {uploading
          ? 'Uploading…'
          : 'Drop an audio file here, or click to browse'}
      </p>
      <p className="text-xs text-muted-foreground/60">mp3 · wav · flac · max 200 MB</p>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Create `frontend/src/pages/HomePage.tsx`**

```tsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AudioUpload } from '@/components/AudioUpload';
import { SongCard } from '@/components/SongCard';
import { listSongs } from '@/lib/api';
import { Song } from '@/types/song';

export function HomePage() {
  const navigate = useNavigate();
  const [songs, setSongs] = useState<Song[]>([]);

  async function refresh() {
    try {
      setSongs(await listSongs());
    } catch { /* ignore */ }
  }

  useEffect(() => { refresh(); }, []);

  function onUploaded(id: string) {
    navigate(`/songs/${id}`);
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-10 space-y-10">
      <div>
        <h1 className="text-3xl font-bold mb-2">Song Analyzer</h1>
        <p className="text-muted-foreground text-sm">
          Upload a track to visualize its structure — stems, sections, BPM and key.
        </p>
      </div>
      <AudioUpload onUploaded={onUploaded} />
      {songs.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-4">Previous analyses</h2>
          <div className="space-y-2">
            {songs.map((s) => (
              <SongCard
                key={s.id}
                song={s}
                onClick={() => navigate(`/songs/${s.id}`)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Create `frontend/src/components/SongCard.tsx`**

```tsx
import { Song } from '@/types/song';

const STATUS_LABEL: Record<string, string> = {
  queued: 'Queued…',
  processing: 'Analyzing…',
  done: 'Done',
  error: 'Error',
};

interface Props {
  song: Song;
  onClick: () => void;
}

export function SongCard({ song, onClick }: Props) {
  return (
    <div
      onClick={onClick}
      className="flex items-center justify-between px-4 py-3 rounded-lg border hover:bg-muted/50 cursor-pointer transition-colors"
    >
      <div className="min-w-0">
        <p className="font-medium truncate">{song.original_name}</p>
        <p className="text-xs text-muted-foreground">
          {song.bpm ? `${song.bpm.toFixed(1)} BPM` : '—'}
          {song.music_key ? ` · ${song.music_key}` : ''}
          {song.duration_sec
            ? ` · ${Math.round(song.duration_sec)}s`
            : ''}
        </p>
      </div>
      <span
        className={`text-xs font-medium px-2 py-1 rounded-full ${
          song.status === 'done'
            ? 'bg-green-100 text-green-700'
            : song.status === 'error'
            ? 'bg-red-100 text-red-700'
            : 'bg-yellow-100 text-yellow-700'
        }`}
      >
        {STATUS_LABEL[song.status] ?? song.status}
      </span>
    </div>
  );
}
```

- [ ] **Step 4: Create `frontend/src/pages/AnalysisPage.tsx` (polling shell)**

```tsx
import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getSong } from '@/lib/api';
import { Song } from '@/types/song';
import { SongTimeline } from '@/components/SongTimeline';

export function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [song, setSong] = useState<Song | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchSong = useCallback(async () => {
    if (!id) return;
    try {
      const data = await getSong(id);
      setSong(data);
    } catch {
      setError('Could not load song.');
    }
  }, [id]);

  useEffect(() => {
    fetchSong();
  }, [fetchSong]);

  // Poll every 2s while not done/error
  useEffect(() => {
    if (!song || song.status === 'done' || song.status === 'error') return;
    const timer = setInterval(fetchSong, 2000);
    return () => clearInterval(timer);
  }, [song, fetchSong]);

  if (error) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-10">
        <p className="text-destructive">{error}</p>
        <button onClick={() => navigate('/')} className="mt-4 underline text-sm">
          Back
        </button>
      </div>
    );
  }

  if (!song) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-10 text-muted-foreground">
        Loading…
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold truncate">{song.original_name}</h1>
          {song.status === 'done' && song.bpm && (
            <p className="text-muted-foreground text-sm mt-1">
              {song.bpm.toFixed(1)} BPM · {song.music_key ?? '?'} ·{' '}
              {song.duration_sec ? `${Math.round(song.duration_sec)}s` : ''}
            </p>
          )}
        </div>
        <button onClick={() => navigate('/')} className="text-sm underline text-muted-foreground">
          Back
        </button>
      </div>

      {(song.status === 'queued' || song.status === 'processing') && (
        <div className="rounded-lg border p-8 text-center text-muted-foreground space-y-2">
          <div className="text-2xl animate-spin inline-block">⟳</div>
          <p>{song.status === 'queued' ? 'Waiting in queue…' : 'Analyzing audio…'}</p>
          <p className="text-xs">This takes 1–3 minutes. This page updates automatically.</p>
        </div>
      )}

      {song.status === 'error' && (
        <div className="rounded-lg border border-destructive p-6 text-destructive">
          Analysis failed: {song.error_message ?? 'unknown error'}
        </div>
      )}

      {song.status === 'done' && song.analysis && (
        <SongTimeline analysis={song.analysis} />
      )}
    </div>
  );
}
```

- [ ] **Step 5: Update `frontend/src/App.tsx`**

```tsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { AnalysisPage } from './pages/AnalysisPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/songs/:id" element={<AnalysisPage />} />
      </Routes>
    </Router>
  );
}

export default App;
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): upload page, analysis page with polling"
```

---

### Task 11: Frontend — SVG Timeline (stems)

**Files:**
- Create: `frontend/src/components/SongTimeline.tsx`
- Create: `frontend/src/components/StemRow.tsx`
- Create: `frontend/src/components/SectionBar.tsx`
- Create: `frontend/src/components/TimeAxis.tsx`

- [ ] **Step 1: Create `frontend/src/components/StemRow.tsx`**

```tsx
import { StemData } from '@/types/song';

const STEM_COLORS: Record<string, string> = {
  vocals: '#6366f1',  // indigo
  drums:  '#f59e0b',  // amber
  bass:   '#10b981',  // emerald
  other:  '#8b5cf6',  // violet
};

interface Props {
  stemName: string;
  data: StemData;
  durationSec: number;
  width: number;
  height: number;
}

export function StemRow({ stemName, data, durationSec, width, height }: Props) {
  const color = STEM_COLORS[stemName] ?? '#94a3b8';
  const envelope = data.envelope;

  if (envelope.length === 0) return null;

  // Build a closed SVG path tracing the energy envelope
  const scaleX = (t: number) => (t / durationSec) * width;
  const scaleY = (e: number) => height - e * height;

  const top = envelope.map(([t, e]) => `${scaleX(t).toFixed(1)},${scaleY(e).toFixed(1)}`).join(' ');

  const pathD =
    `M 0,${height} ` +
    envelope.map(([t, e]) => `L ${scaleX(t).toFixed(1)},${scaleY(e).toFixed(1)}`).join(' ') +
    ` L ${width},${height} Z`;

  return (
    <svg width={width} height={height} className="block">
      <path d={pathD} fill={color} opacity={0.75} />
    </svg>
  );
}
```

- [ ] **Step 2: Create `frontend/src/components/SectionBar.tsx`**

```tsx
import { Section } from '@/types/song';

interface Props {
  sections: Section[];
  durationSec: number;
  width: number;
  height: number;
  labels: Record<string, string>; // label → custom name
  onRename: (label: string, name: string) => void;
}

const SECTION_COLORS = [
  '#cbd5e1', '#94a3b8', '#64748b', '#475569', '#334155',
];

export function SectionBar({ sections, durationSec, width, height, labels, onRename }: Props) {
  if (sections.length === 0) {
    return <div style={{ width, height }} className="bg-muted/30 rounded" />;
  }

  return (
    <svg width={width} height={height} className="block">
      {sections.map((sec, i) => {
        const x = (sec.start_sec / durationSec) * width;
        const w = ((sec.end_sec - sec.start_sec) / durationSec) * width;
        const displayName = labels[sec.label] ?? sec.label;
        return (
          <g key={sec.label}>
            <rect
              x={x}
              y={0}
              width={w}
              height={height}
              fill={SECTION_COLORS[i % SECTION_COLORS.length]}
              rx={2}
            />
            {w > 24 && (
              <text
                x={x + 6}
                y={height / 2 + 4}
                fontSize={11}
                fill="#1e293b"
                className="select-none"
              >
                {displayName}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
```

- [ ] **Step 3: Create `frontend/src/components/TimeAxis.tsx`**

```tsx
interface Props {
  durationSec: number;
  bpm: number;
  width: number;
  height: number;
  mode: 'seconds' | 'bars';
}

export function TimeAxis({ durationSec, bpm, width, height, mode }: Props) {
  const ticks: { x: number; label: string }[] = [];

  if (mode === 'seconds') {
    const step = durationSec > 120 ? 30 : durationSec > 60 ? 10 : 5;
    for (let t = 0; t <= durationSec; t += step) {
      ticks.push({ x: (t / durationSec) * width, label: `${t}s` });
    }
  } else {
    // bars mode
    const secPerBeat = 60 / (bpm || 120);
    const secPerBar = secPerBeat * 4;
    const totalBars = Math.floor(durationSec / secPerBar);
    const step = totalBars > 64 ? 8 : totalBars > 32 ? 4 : 2;
    for (let bar = 0; bar <= totalBars; bar += step) {
      const t = bar * secPerBar;
      ticks.push({ x: (t / durationSec) * width, label: `${bar + 1}` });
    }
  }

  return (
    <svg width={width} height={height} className="block">
      <line x1={0} y1={0} x2={width} y2={0} stroke="#e2e8f0" strokeWidth={1} />
      {ticks.map(({ x, label }) => (
        <g key={label}>
          <line x1={x} y1={0} x2={x} y2={6} stroke="#94a3b8" strokeWidth={1} />
          <text x={x + 2} y={height - 2} fontSize={9} fill="#94a3b8">
            {label}
          </text>
        </g>
      ))}
    </svg>
  );
}
```

- [ ] **Step 4: Create `frontend/src/components/SongTimeline.tsx`**

```tsx
import { useState } from 'react';
import { AnalysisResult, Section } from '@/types/song';
import { StemRow } from './StemRow';
import { SectionBar } from './SectionBar';
import { TimeAxis } from './TimeAxis';

const STEM_ORDER = ['vocals', 'drums', 'bass', 'other'];
const ROW_HEIGHT = 64;
const SECTION_HEIGHT = 28;
const AXIS_HEIGHT = 24;
const LABEL_WIDTH = 72;

interface Props {
  analysis: AnalysisResult;
}

export function SongTimeline({ analysis }: Props) {
  const [axisMode, setAxisMode] = useState<'seconds' | 'bars'>('bars');
  const [sectionLabels, setSectionLabels] = useState<Record<string, string>>({});
  const [editingSection, setEditingSection] = useState<string | null>(null);

  const timelineWidth = Math.max(600, Math.min(1100, window.innerWidth - LABEL_WIDTH - 80));

  function handleRename(label: string, name: string) {
    setSectionLabels((prev) => ({ ...prev, [label]: name }));
    setEditingSection(null);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-sm text-muted-foreground">Axis:</span>
        <button
          onClick={() => setAxisMode(axisMode === 'bars' ? 'seconds' : 'bars')}
          className="text-sm px-3 py-1 rounded border hover:bg-muted transition-colors"
        >
          {axisMode === 'bars' ? 'Bars' : 'Seconds'}
        </button>
      </div>

      <div className="overflow-x-auto">
        <div style={{ minWidth: LABEL_WIDTH + timelineWidth }}>
          {/* Section bar */}
          <div className="flex items-center">
            <div style={{ width: LABEL_WIDTH }} className="text-xs text-muted-foreground pr-2 text-right">
              Sections
            </div>
            <SectionBar
              sections={analysis.sections}
              durationSec={analysis.duration_sec}
              width={timelineWidth}
              height={SECTION_HEIGHT}
              labels={sectionLabels}
              onRename={handleRename}
            />
          </div>

          {/* Stem rows */}
          {STEM_ORDER.filter((s) => analysis.stems[s]).map((stemName) => (
            <div key={stemName} className="flex items-center mt-1">
              <div
                style={{ width: LABEL_WIDTH }}
                className="text-xs font-medium capitalize text-right pr-2 text-muted-foreground"
              >
                {stemName}
              </div>
              <div className="rounded overflow-hidden bg-muted/20">
                <StemRow
                  stemName={stemName}
                  data={analysis.stems[stemName]}
                  durationSec={analysis.duration_sec}
                  width={timelineWidth}
                  height={ROW_HEIGHT}
                />
              </div>
            </div>
          ))}

          {/* Time axis */}
          <div className="flex items-center mt-1">
            <div style={{ width: LABEL_WIDTH }} />
            <TimeAxis
              durationSec={analysis.duration_sec}
              bpm={analysis.bpm}
              width={timelineWidth}
              height={AXIS_HEIGHT}
              mode={axisMode}
            />
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-4 flex-wrap text-xs text-muted-foreground">
        {STEM_ORDER.map((s) => (
          <span key={s} className="flex items-center gap-1">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{
                background: { vocals: '#6366f1', drums: '#f59e0b', bass: '#10b981', other: '#8b5cf6' }[s],
              }}
            />
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Start dev server and verify it renders**

```bash
cd frontend && npm run dev
```

Open http://localhost:3000, verify the app loads without console errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/SongTimeline.tsx frontend/src/components/StemRow.tsx frontend/src/components/SectionBar.tsx frontend/src/components/TimeAxis.tsx
git commit -m "feat(frontend): SVG timeline with stem rows, section bar, time axis"
```

---

## Phase 2 — BPM, Key & Structural Segmentation

### Task 12: Audio Service — BPM + beat grid

**Files:**
- Create: `audio-service/src/analysis/beat_analysis.py`
- Create: `audio-service/tests/test_beat_analysis.py`
- Modify: `audio-service/src/main.py`

- [ ] **Step 1: Create `audio-service/src/analysis/beat_analysis.py`**

```python
import numpy as np
import librosa

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key-finding profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_bpm_and_beats(audio: np.ndarray, sr: int) -> tuple[float, list[float]]:
    """Return (bpm, beat_times_in_seconds)."""
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    return round(bpm, 2), [round(t, 4) for t in beat_times]


def detect_key(audio: np.ndarray, sr: int) -> str:
    """Estimate musical key using Krumhansl-Schmuckler profiles on chroma_cqt."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # shape (12,)

    best_score = -np.inf
    best_key = 'C major'

    for i in range(12):
        major_score = np.corrcoef(chroma_mean, np.roll(_MAJOR, i))[0, 1]
        minor_score = np.corrcoef(chroma_mean, np.roll(_MINOR, i))[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = f'{NOTE_NAMES[i]} major'
        if minor_score > best_score:
            best_score = minor_score
            best_key = f'{NOTE_NAMES[i]} minor'

    return best_key
```

- [ ] **Step 2: Create `audio-service/tests/test_beat_analysis.py`**

```python
import numpy as np
import pytest
from src.analysis.beat_analysis import detect_bpm_and_beats, detect_key


def test_bpm_returns_positive_float(sample_audio):
    y, sr = sample_audio
    bpm, beats = detect_bpm_and_beats(y, sr)
    assert isinstance(bpm, float)
    assert bpm > 0
    assert isinstance(beats, list)


def test_beats_are_sorted(sample_audio):
    y, sr = sample_audio
    _, beats = detect_bpm_and_beats(y, sr)
    assert beats == sorted(beats)


def test_key_returns_valid_string(sample_audio):
    y, sr = sample_audio
    key = detect_key(y, sr)
    assert isinstance(key, str)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    assert any(key.startswith(n) for n in notes)
    assert key.endswith('major') or key.endswith('minor')
```

- [ ] **Step 3: Run tests**

```bash
cd audio-service && python -m pytest tests/test_beat_analysis.py -v
```

Expected: `3 passed`

- [ ] **Step 4: Commit**

```bash
git add audio-service/src/analysis/beat_analysis.py audio-service/tests/test_beat_analysis.py
git commit -m "feat(audio-service): BPM, beat grid, and key detection"
```

---

### Task 13: Audio Service — Structural Segmentation

**Files:**
- Create: `audio-service/src/analysis/segmentation.py`
- Modify: `audio-service/src/main.py`

- [ ] **Step 1: Create `audio-service/src/analysis/segmentation.py`**

```python
import numpy as np
import librosa
import string


def _uniform_sections(duration_sec: float, n: int = 4) -> list[dict]:
    """Fallback: divide song into n equal sections labeled A, B, C…"""
    seg_dur = duration_sec / n
    labels = list(string.ascii_uppercase)
    return [
        {
            "label": labels[i],
            "start_sec": round(i * seg_dur, 3),
            "end_sec": round((i + 1) * seg_dur, 3),
        }
        for i in range(n)
    ]


def detect_sections(audio: np.ndarray, sr: int, duration_sec: float) -> list[dict]:
    """
    Use MSAF if available; fall back to uniform segmentation.
    MSAF expects an audio file path, so we write a temp file when using it.
    """
    try:
        import msaf
        import tempfile
        import soundfile as sf
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, sr)
            boundaries, labels = msaf.process(tmp_path, boundaries_id="sf", labels_id=None)
            # boundaries is in seconds; labels is array of ints
            sections = []
            label_chars = list(string.ascii_uppercase)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                sections.append({
                    "label": label_chars[i % len(label_chars)],
                    "start_sec": round(float(start), 3),
                    "end_sec": round(float(end), 3),
                })
            return sections if sections else _uniform_sections(duration_sec)
        finally:
            os.unlink(tmp_path)
    except Exception:
        # MSAF not installed or failed — use uniform fallback
        return _uniform_sections(duration_sec)
```

- [ ] **Step 2: Update `audio-service/src/main.py` to wire in beat analysis and segmentation**

```python
import os
import tempfile
import traceback

from fastapi import FastAPI, HTTPException
from .models import AnalyzeRequest, AnalysisResult
from .storage import download_to_path
from .analysis.demucs_runner import separate_stems
from .analysis.envelope import compute_envelope
from .analysis.beat_analysis import detect_bpm_and_beats, detect_key
from .analysis.segmentation import detect_sections
import librosa

app = FastAPI(title="Audio Analysis Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.audio")

        try:
            download_to_path(req.storage_key, input_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Cannot fetch audio: {e}")

        try:
            stems_data = separate_stems(input_path, tmpdir)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Demucs failed: {e}")

        # Mix all stems to mono for global analysis
        all_stems = [audio for audio, _ in stems_data.values()]
        sr = next(iter(stems_data.values()))[1]
        mix_mono = np.mean(all_stems, axis=0) if all_stems else np.zeros(1)

        duration_sec = len(mix_mono) / sr
        bpm, beat_grid = detect_bpm_and_beats(mix_mono, sr)
        key = detect_key(mix_mono, sr)
        sections = detect_sections(mix_mono, sr, duration_sec)

        stems: dict = {}
        for stem_name, (audio, stem_sr) in stems_data.items():
            stems[stem_name] = {"envelope": compute_envelope(audio, stem_sr)}

        return {
            "bpm": bpm,
            "key": key,
            "duration_sec": round(duration_sec, 3),
            "beat_grid": beat_grid,
            "sections": sections,
            "stems": stems,
        }
```

Add `import numpy as np` at the top of main.py.

- [ ] **Step 3: Run all audio-service tests**

```bash
cd audio-service && python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add audio-service/src/analysis/segmentation.py audio-service/src/main.py
git commit -m "feat(audio-service): BPM/key/segmentation wired into /analyze"
```

---

### Task 14: Frontend — Section bar renaming + bars axis

**Files:**
- Modify: `frontend/src/components/SectionBar.tsx`
- Modify: `frontend/src/components/SongTimeline.tsx`

- [ ] **Step 1: Add inline rename to `SectionBar.tsx`**

Replace the `<text>` element with a `<foreignObject>` that shows an input on double-click:

```tsx
import { Section } from '@/types/song';
import { useState } from 'react';

const SECTION_COLORS = [
  '#cbd5e1', '#94a3b8', '#64748b', '#475569', '#334155',
];

interface Props {
  sections: Section[];
  durationSec: number;
  width: number;
  height: number;
  labels: Record<string, string>;
  onRename: (label: string, name: string) => void;
}

export function SectionBar({ sections, durationSec, width, height, labels, onRename }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');

  if (sections.length === 0) {
    return <div style={{ width, height }} className="bg-muted/30 rounded" />;
  }

  function startEdit(label: string) {
    setEditing(label);
    setDraft(labels[label] ?? label);
  }

  function commitEdit() {
    if (editing) onRename(editing, draft.trim() || editing);
    setEditing(null);
  }

  return (
    <svg width={width} height={height} className="block">
      {sections.map((sec, i) => {
        const x = (sec.start_sec / durationSec) * width;
        const w = ((sec.end_sec - sec.start_sec) / durationSec) * width;
        const displayName = labels[sec.label] ?? sec.label;
        const isEditing = editing === sec.label;
        return (
          <g key={sec.label} onDoubleClick={() => startEdit(sec.label)} style={{ cursor: 'pointer' }}>
            <rect x={x} y={0} width={w} height={height} fill={SECTION_COLORS[i % SECTION_COLORS.length]} rx={2} />
            {w > 24 && !isEditing && (
              <text x={x + 6} y={height / 2 + 4} fontSize={11} fill="#1e293b" className="select-none">
                {displayName}
              </text>
            )}
            {isEditing && w > 40 && (
              <foreignObject x={x + 2} y={2} width={w - 4} height={height - 4}>
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commitEdit}
                  onKeyDown={(e) => e.key === 'Enter' && commitEdit()}
                  style={{ width: '100%', height: '100%', fontSize: 11, border: 'none', background: 'transparent', outline: 'none' }}
                />
              </foreignObject>
            )}
          </g>
        );
      })}
    </svg>
  );
}
```

- [ ] **Step 2: Add rename instruction text to `SongTimeline.tsx`**

Below the axis mode toggle, add:
```tsx
<span className="text-xs text-muted-foreground">
  Double-click a section to rename it.
</span>
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/SectionBar.tsx frontend/src/components/SongTimeline.tsx
git commit -m "feat(frontend): section inline renaming + bars axis"
```

---

## Phase 3 — Polish

### Task 15: Backend — DELETE endpoint + error handling

- [ ] **Step 1: Verify DELETE /api/songs/:id works end-to-end**

```bash
# Upload a test file, get id, then delete
curl -X DELETE http://localhost:5001/api/songs/<id>
```

Expected: `204 No Content`

- [ ] **Step 2: Add multer error handling middleware to `backend/src/routes/songs.ts`**

After the last route, before `export default router`:

```typescript
// Handle multer file-too-large error
router.use((err: Error, _req: Request, res: Response, next: NextFunction) => {
  if (err.message?.includes('File too large') || err.name === 'MulterError') {
    res.status(400).json({ error: `File too large (max ${MAX_BYTES / 1024 / 1024} MB)` });
    return;
  }
  next(err);
});
```

- [ ] **Step 3: Add delete button to `SongCard.tsx`**

```tsx
import { deleteSong } from '@/lib/api';

// Add onDelete prop:
interface Props {
  song: Song;
  onClick: () => void;
  onDelete?: () => void;
}

export function SongCard({ song, onClick, onDelete }: Props) {
  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm(`Delete "${song.original_name}"?`)) return;
    await deleteSong(song.id).catch(() => {});
    onDelete?.();
  }

  return (
    <div
      onClick={onClick}
      className="flex items-center justify-between px-4 py-3 rounded-lg border hover:bg-muted/50 cursor-pointer transition-colors"
    >
      <div className="min-w-0">
        <p className="font-medium truncate">{song.original_name}</p>
        <p className="text-xs text-muted-foreground">
          {song.bpm ? `${song.bpm.toFixed(1)} BPM` : '—'}
          {song.music_key ? ` · ${song.music_key}` : ''}
          {song.duration_sec ? ` · ${Math.round(song.duration_sec)}s` : ''}
        </p>
      </div>
      <div className="flex items-center gap-2">
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${
          song.status === 'done' ? 'bg-green-100 text-green-700'
          : song.status === 'error' ? 'bg-red-100 text-red-700'
          : 'bg-yellow-100 text-yellow-700'
        }`}>
          {STATUS_LABEL[song.status] ?? song.status}
        </span>
        {onDelete && (
          <button
            onClick={handleDelete}
            className="text-muted-foreground hover:text-destructive transition-colors p-1"
            title="Delete"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
```

Update `HomePage.tsx` to pass `onDelete`:
```tsx
<SongCard key={s.id} song={s} onClick={() => navigate(`/songs/${s.id}`)} onDelete={refresh} />
```

- [ ] **Step 4: Commit**

```bash
git add backend/src/routes/songs.ts frontend/src/components/SongCard.tsx frontend/src/pages/HomePage.tsx
git commit -m "feat: delete song + multer error handling"
```

---

### Task 16: End-to-End verification + docker-compose smoke test

- [ ] **Step 1: Copy backend env**

```bash
cp backend/env.example backend/.env
```

- [ ] **Step 2: Start full stack**

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Wait for all healthchecks to pass.

- [ ] **Step 3: Verify services**

```bash
curl http://localhost:5001/api/health     # {"status":"ok"}
curl http://localhost:8000/health         # {"status":"ok"}
curl http://localhost:9000/minio/health/live  # ok
```

- [ ] **Step 4: Upload a real mp3 and verify full pipeline**

1. Open http://localhost:3000
2. Upload an mp3 file
3. Watch `processing` → `done`
4. Verify timeline renders with 4 stem rows
5. Verify BPM, key, duration appear
6. Verify sections appear (may be uniform A/B/C/D if MSAF unavailable)
7. Double-click a section to rename it
8. Toggle axis between Bars / Seconds
9. Delete the song from the list

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "feat: MVP complete — song structure analyzer"
```

---

## Spec Coverage Check

| Requirement | Task |
|---|---|
| Upload mp3/wav/flac with magic-byte validation | Task 5 |
| MinIO object storage (not disk) | Tasks 1, 3 |
| Demucs 4-stem separation | Task 6 |
| RMS envelope per stem (20–50 pts/sec) | Task 6 |
| BPM detection | Task 12 |
| Key detection (Krumhansl-Schmuckler) | Task 12 |
| Beat grid | Task 12 |
| Structural segmentation (MSAF + fallback) | Task 13 |
| Postgres schema (songs, song_jobs, song_analysis) | Task 4 |
| Job queue with FOR UPDATE SKIP LOCKED | Task 8 |
| POST /api/songs | Task 5 |
| GET /api/songs/:id (with analysis when done) | Task 5 |
| GET /api/songs list | Task 5 |
| DELETE /api/songs/:id + storage cleanup | Tasks 5, 15 |
| Frontend polling until done | Task 10 |
| SVG timeline with 4 stem rows | Task 11 |
| Section bar above stems | Task 11 |
| Section renaming | Task 14 |
| Axis toggle (bars / seconds) | Task 11 |
| BPM/key display | Task 10 |
| Previous analyses list with delete | Task 10, 15 |
| Demucs model cached in Docker volume | Task 1 (docker-compose volume) |
| Config from env vars only | Tasks 1, 3, 5 |
| File size limit (200 MB) + format whitelist | Task 5 |
| No disk uploads (MinIO only) | Tasks 3, 5 |
| Audio service stateless, no DB access | Tasks 6–7, 13 |
| Phase 2 door open (events) in data model | schema has result_json JSONB — add fields later |
