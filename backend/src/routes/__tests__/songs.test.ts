import request from 'supertest';
import app from '../../app.js';

jest.mock('file-type', () => ({
  fileTypeFromBuffer: jest.fn().mockResolvedValue({ mime: 'audio/mpeg', ext: 'mp3' }),
}));

jest.mock('../../services/storageService.js', () => ({
  uploadFile: jest.fn().mockResolvedValue(undefined),
  deleteFile: jest.fn().mockResolvedValue(undefined),
  deleteFilesByPrefix: jest.fn().mockResolvedValue(undefined),
  ensureBucketExists: jest.fn().mockResolvedValue(undefined),
  presignDownload: jest.fn().mockResolvedValue('https://example.com/default'),
}));

jest.mock('../../services/songService.js', () => {
  const mockSong = {
    id: 'test-uuid',
    status: 'queued',
    storage_key: 'songs/test.mp3',
    original_name: 'test.mp3',
    created_at: new Date().toISOString(),
    bpm: null,
    music_key: null,
    duration_sec: null,
    error_message: null,
  };
  return {
    createSong: jest.fn().mockResolvedValue(mockSong),
    createJob: jest.fn().mockResolvedValue('job-id'),
    listSongs: jest.fn().mockResolvedValue([mockSong]),
    getSong: jest.fn().mockResolvedValue(null),
    deleteSong: jest.fn().mockResolvedValue(null),
    getSongAnalysis: jest.fn().mockResolvedValue(null),
  };
});

describe('GET /api/songs', () => {
  it('returns list of songs', async () => {
    const res = await request(app).get('/api/songs');
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body)).toBe(true);
    expect(res.body).toHaveLength(1);
    expect(res.body[0].id).toBe('test-uuid');
  });
});

describe('GET /api/songs/:id', () => {
  it('returns 404 for unknown id', async () => {
    const res = await request(app).get('/api/songs/nonexistent');
    expect(res.status).toBe(404);
    expect(res.body.error).toBe('Song not found');
  });
});

describe('DELETE /api/songs/:id', () => {
  it('returns 404 for unknown id', async () => {
    const res = await request(app).delete('/api/songs/nonexistent');
    expect(res.status).toBe(404);
    expect(res.body.error).toBe('Song not found');
  });
});

describe('GET /api/songs/:id/audio', () => {
  it('returns 302 redirect to presigned URL', async () => {
    const songService = await import('../../services/songService.js');
    (songService.getSong as jest.Mock).mockResolvedValueOnce({
      id: 'song-1',
      storage_key: 'songs/song-1.mp3',
      original_name: 't.mp3',
      status: 'done',
      duration_sec: 1,
      bpm: 1,
      music_key: 'C major',
      error_message: null,
      created_at: new Date().toISOString(),
    });

    const storage = await import('../../services/storageService.js');
    (storage.presignDownload as jest.Mock).mockResolvedValueOnce(
      'https://example.com/audio'
    );

    const res = await request(app).get('/api/songs/song-1/audio');
    expect(res.status).toBe(302);
    expect(res.headers.location).toBe('https://example.com/audio');
  });

  it('returns 404 for unknown song', async () => {
    const songService = await import('../../services/songService.js');
    (songService.getSong as jest.Mock).mockResolvedValueOnce(null);
    const res = await request(app).get('/api/songs/nope/audio');
    expect(res.status).toBe(404);
  });
});
