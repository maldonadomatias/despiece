jest.mock('@aws-sdk/client-s3', () => ({
  S3Client: jest.fn().mockImplementation(() => ({
    send: jest.fn().mockResolvedValue({}),
  })),
  PutObjectCommand: jest.fn(),
  DeleteObjectCommand: jest.fn(),
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
