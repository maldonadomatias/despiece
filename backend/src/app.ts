import express, { Express, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import path from 'path';
import { errorHandler } from './utils/errorHandler.js';
import songsRouter from './routes/songs.js';

const app: Express = express();

// CORS configuration
const allowedOrigins = [
  process.env.FRONTEND_URL || 'http://localhost:3000',
].filter(Boolean);

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin)) {
        callback(null, true);
      } else if (
        process.env.NODE_ENV === 'development' &&
        origin.startsWith('http://localhost:')
      ) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Serve static files
app.use('/uploads', express.static(path.join(process.cwd(), 'uploads')));

// Health check route
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// API routes
app.use('/api/songs', songsRouter);

// Handle multer file-upload errors before the generic error handler
app.use((err: Error & { code?: string }, _req: Request, res: Response, next: NextFunction) => {
  if (
    err.constructor?.name === 'MulterError' ||
    err.code === 'LIMIT_FILE_SIZE' ||
    err.message?.startsWith('Only mp3')
  ) {
    res.status(400).json({ error: err.message ?? 'File upload error' });
    return;
  }
  next(err);
});

// Error handler (must be last)
app.use(errorHandler);

export default app;
