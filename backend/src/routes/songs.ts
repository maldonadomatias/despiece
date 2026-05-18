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
