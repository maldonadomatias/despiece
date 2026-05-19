import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  DeleteObjectsCommand,
  ListObjectsV2Command,
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

let _s3: S3Client | null = null;
function getS3(): S3Client {
  if (!_s3) _s3 = buildS3Client();
  return _s3;
}

function getBucket(): string {
  return process.env.S3_BUCKET ?? 'songs';
}

export async function ensureBucketExists(): Promise<void> {
  const s3 = getS3();
  const BUCKET = getBucket();
  try {
    await s3.send(new HeadBucketCommand({ Bucket: BUCKET }));
  } catch (headErr: unknown) {
    const name = (headErr as { name?: string }).name;
    if (name !== 'NoSuchBucket' && name !== 'NotFound' && name !== '404') {
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
    }
  }
}

export async function uploadFile(
  key: string,
  buffer: Buffer,
  contentType: string
): Promise<void> {
  try {
    await getS3().send(
      new PutObjectCommand({
        Bucket: getBucket(),
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
    await getS3().send(new DeleteObjectCommand({ Bucket: getBucket(), Key: key }));
  } catch (err) {
    logger.error({ key, err }, 'Failed to delete file from S3');
    throw err;
  }
}

export async function deleteFilesByPrefix(prefix: string): Promise<void> {
  const s3 = getS3();
  const bucket = getBucket();
  let continuationToken: string | undefined;

  do {
    const listResp = await s3.send(
      new ListObjectsV2Command({
        Bucket: bucket,
        Prefix: prefix,
        ContinuationToken: continuationToken,
      })
    );

    const objects = (listResp.Contents ?? [])
      .map((o) => o.Key)
      .filter((k): k is string => typeof k === 'string')
      .map((Key) => ({ Key }));

    if (objects.length > 0) {
      await s3.send(
        new DeleteObjectsCommand({
          Bucket: bucket,
          Delete: { Objects: objects },
        })
      );
    }

    continuationToken = listResp.IsTruncated ? listResp.NextContinuationToken : undefined;
  } while (continuationToken);
}
