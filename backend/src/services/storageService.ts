import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  HeadBucketCommand,
  CreateBucketCommand,
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
    forcePathStyle: true,
  });
}

const s3 = buildS3Client();
const BUCKET = process.env.S3_BUCKET ?? 'songs';

export async function ensureBucketExists(): Promise<void> {
  try {
    await s3.send(new HeadBucketCommand({ Bucket: BUCKET }));
  } catch (headErr: unknown) {
    const name = (headErr as { name?: string }).name;
    if (name !== 'NoSuchBucket' && name !== 'NotFound' && name !== '404') {
      // Bucket may exist but we lack permission to HEAD it — proceed
      logger.warn({ err: headErr }, 'HeadBucket check failed; assuming bucket exists');
      return;
    }
    try {
      await s3.send(new CreateBucketCommand({ Bucket: BUCKET }));
      logger.info({ bucket: BUCKET }, 'Created S3 bucket');
    } catch (createErr: unknown) {
      const createName = (createErr as { name?: string }).name;
      if (
        createName !== 'BucketAlreadyOwnedByYou' &&
        createName !== 'BucketAlreadyExists'
      ) {
        throw createErr;
      }
      // Another instance already created it — fine
    }
  }
}

export async function uploadFile(
  key: string,
  buffer: Buffer,
  contentType: string
): Promise<void> {
  try {
    await s3.send(
      new PutObjectCommand({
        Bucket: BUCKET,
        Key: key,
        Body: buffer,
        ContentType: contentType,
      })
    );
  } catch (err) {
    logger.error({ key, err }, 'Failed to upload file to S3');
    throw err;
  }
}

export async function deleteFile(key: string): Promise<void> {
  try {
    await s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: key }));
  } catch (err) {
    logger.error({ key, err }, 'Failed to delete file from S3');
    throw err;
  }
}
