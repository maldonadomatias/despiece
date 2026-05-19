const mockSend = jest.fn().mockResolvedValue({});

jest.mock('@aws-sdk/s3-request-presigner', () => ({
  getSignedUrl: jest.fn().mockResolvedValue('https://example.com/signed'),
}));

jest.mock('@aws-sdk/client-s3', () => ({
  S3Client: jest.fn().mockImplementation(() => ({
    send: mockSend,
  })),
  GetObjectCommand: jest.fn().mockImplementation((input) => ({ input, constructor: { name: 'GetObjectCommand' } })),
  PutObjectCommand: jest.fn().mockImplementation((input) => ({ input, constructor: { name: 'PutObjectCommand' } })),
  DeleteObjectCommand: jest.fn().mockImplementation((input) => ({ input, constructor: { name: 'DeleteObjectCommand' } })),
  DeleteObjectsCommand: jest.fn().mockImplementation((input) => ({ input, constructor: { name: 'DeleteObjectsCommand' } })),
  ListObjectsV2Command: jest.fn().mockImplementation((input) => ({ input, constructor: { name: 'ListObjectsV2Command' } })),
  HeadBucketCommand: jest.fn().mockImplementation((input) => ({ input, constructor: { name: 'HeadBucketCommand' } })),
  CreateBucketCommand: jest.fn().mockImplementation((input) => ({ input, constructor: { name: 'CreateBucketCommand' } })),
}));

import { uploadFile, deleteFile, deleteFilesByPrefix, presignDownload } from '../storageService.js';

beforeEach(() => {
  mockSend.mockResolvedValue({});
});

describe('uploadFile', () => {
  it('sends PutObjectCommand with correct key and contentType', async () => {
    await uploadFile('test/song.mp3', Buffer.from('data'), 'audio/mpeg');
    expect(mockSend).toHaveBeenCalledTimes(1);
    const cmd = mockSend.mock.calls[0][0];
    expect(cmd.input).toMatchObject({
      Key: 'test/song.mp3',
      ContentType: 'audio/mpeg',
    });
  });

  it('rethrows on S3 error', async () => {
    mockSend.mockRejectedValueOnce(new Error('AccessDenied'));
    await expect(
      uploadFile('test/song.mp3', Buffer.from('data'), 'audio/mpeg')
    ).rejects.toThrow('AccessDenied');
  });
});

describe('deleteFile', () => {
  it('sends DeleteObjectCommand with correct key', async () => {
    await deleteFile('test/song.mp3');
    const cmd = mockSend.mock.calls[0][0];
    expect(cmd.input).toMatchObject({ Key: 'test/song.mp3' });
  });

  it('rethrows on S3 error', async () => {
    mockSend.mockRejectedValueOnce(new Error('NoSuchKey'));
    await expect(deleteFile('test/song.mp3')).rejects.toThrow('NoSuchKey');
  });
});

describe('deleteFilesByPrefix', () => {
  it('lists then batch-deletes when objects exist', async () => {
    const { ListObjectsV2Command, DeleteObjectsCommand } = await import('@aws-sdk/client-s3');
    const listMock = ListObjectsV2Command as unknown as jest.Mock;
    const deleteMock = DeleteObjectsCommand as unknown as jest.Mock;
    listMock.mockClear();
    deleteMock.mockClear();
    mockSend.mockReset();

    // First call: list returns two objects; second call: delete succeeds
    mockSend
      .mockResolvedValueOnce({
        Contents: [
          { Key: 'songs/abc/stems/vocals.mp3' },
          { Key: 'songs/abc/stems/drums.mp3' },
        ],
        IsTruncated: false,
      })
      .mockResolvedValueOnce({ Deleted: [{ Key: 'songs/abc/stems/vocals.mp3' }] });

    await deleteFilesByPrefix('songs/abc/');

    expect(listMock).toHaveBeenCalledTimes(1);
    expect(listMock).toHaveBeenCalledWith(
      expect.objectContaining({ Prefix: 'songs/abc/' })
    );
    expect(deleteMock).toHaveBeenCalledTimes(1);
    expect(deleteMock).toHaveBeenCalledWith(
      expect.objectContaining({
        Delete: {
          Objects: [
            { Key: 'songs/abc/stems/vocals.mp3' },
            { Key: 'songs/abc/stems/drums.mp3' },
          ],
        },
      })
    );
  });

  it('skips delete when no objects match prefix', async () => {
    const { ListObjectsV2Command, DeleteObjectsCommand } = await import('@aws-sdk/client-s3');
    const listMock = ListObjectsV2Command as unknown as jest.Mock;
    const deleteMock = DeleteObjectsCommand as unknown as jest.Mock;
    listMock.mockClear();
    deleteMock.mockClear();
    mockSend.mockReset();

    mockSend.mockResolvedValueOnce({ Contents: [], IsTruncated: false });

    await deleteFilesByPrefix('songs/missing/');

    expect(listMock).toHaveBeenCalledTimes(1);
    expect(deleteMock).not.toHaveBeenCalled();
  });
});

describe('presignDownload', () => {
  it('returns a signed URL', async () => {
    const url = await presignDownload('songs/abc/stems/vocals.mp3', 60);
    expect(url).toBe('https://example.com/signed');
  });
});
