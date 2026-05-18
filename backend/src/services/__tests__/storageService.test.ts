const mockSend = jest.fn().mockResolvedValue({});

jest.mock('@aws-sdk/client-s3', () => ({
  S3Client: jest.fn().mockImplementation(() => ({
    send: mockSend,
  })),
  PutObjectCommand: jest.fn().mockImplementation((input) => ({ input })),
  DeleteObjectCommand: jest.fn().mockImplementation((input) => ({ input })),
  HeadBucketCommand: jest.fn().mockImplementation((input) => ({ input })),
  CreateBucketCommand: jest.fn().mockImplementation((input) => ({ input })),
}));

import { uploadFile, deleteFile } from '../storageService.js';

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
