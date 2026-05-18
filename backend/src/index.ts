import dotenv from 'dotenv';
import app from './app.js';
import logger from './utils/logger.js';
import { ensureBucketExists } from './services/storageService.js';
import { startJobWorker } from './workers/jobWorker.js';

dotenv.config();

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
