# Project: nodejs-react-starter

A modern full-stack web application template with React 19, Node.js 20, Express, TypeScript, PostgreSQL 15, and Docker.

## Tech Stack

### Frontend
- **React 19** - Latest React with modern features
- **Vite 6** - Build tool and dev server
- **TypeScript 5.7** - Type-safe development
- **Tailwind CSS 4** - Utility-first CSS
- **shadcn/ui** - UI components (radix-nova style, lucide icons)
- **React Router 7** - Client-side routing
- **Vitest** - Unit testing
- **Axios** - HTTP client

### Backend
- **Node.js 20+** - JavaScript runtime
- **Express 4** - Web framework
- **TypeScript 5.7** - Type-safe development
- **PostgreSQL 15** - Relational database (pg driver)
- **Jest** - Testing framework
- **Pino** - Fast logger
- **Zod** - Schema validation
- **tsx** - TypeScript execution in dev

### DevOps
- **Docker & Docker Compose** - Containerization
- **ESLint 9** - Code linting
- **Prettier** - Code formatting

## Project Structure

```
├── backend/                 # Node.js + Express backend
│   ├── src/
│   │   ├── app.ts          # Express app configuration
│   │   ├── index.ts        # Entry point
│   │   ├── config/         # Configuration
│   │   ├── db/             # Database (connect.ts, migrate.ts)
│   │   ├── domain/         # Domain models/types
│   │   ├── middleware/     # Express middleware
│   │   ├── routes/         # API route handlers
│   │   ├── services/       # Business logic
│   │   ├── utils/          # Utilities (errorHandler, logger)
│   │   └── workers/        # Background workers
│   ├── uploads/            # File uploads
│   └── package.json
├── frontend/                # React frontend
│   ├── src/
│   │   ├── main.jsx        # Entry point
│   │   ├── App.tsx         # Root component
│   │   ├── components/     # React components (incl. ui/ for shadcn)
│   │   ├── pages/          # Page components
│   │   ├── lib/            # Utils and API client
│   │   └── types/          # TypeScript types
│   └── package.json
├── docker-compose.yml
├── docker-compose.dev.yml
└── docker-compose.prod.yml
```

## Build, Test, and Run Commands

### Root (from project root)

| Command | Description |
|---------|-------------|
| `npm run start:dev` | Start PostgreSQL via Docker + frontend + backend in dev mode |
| `npm run install:all` | Install dependencies for frontend and backend |
| `npm run build` | Install + build frontend and backend |
| `npm run lint` | Lint frontend and backend |
| `npm run test` | Run tests for frontend and backend |
| `npm run format` | Format code with Prettier |
| `npm run format:check` | Check formatting without changes |

### Backend (`cd backend`)

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server with tsx watch |
| `npm run build` | Compile TypeScript to `dist/` |
| `npm run start` | Run production server (`node dist/index.js`) |
| `npm run db:migrate` | Run database migrations |
| `npm run test` | Run Jest tests |
| `npm run test:watch` | Jest in watch mode |
| `npm run test:coverage` | Jest with coverage |
| `npm run lint` | ESLint |
| `npm run format` / `format:check` | Prettier |

### Frontend (`cd frontend`)

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run test` | Run Vitest |
| `npm run test:ui` | Vitest with UI |
| `npm run test:coverage` | Vitest with coverage |
| `npm run lint` | ESLint |
| `npm run format` / `format:check` | Prettier |

### Docker

```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### URLs (dev)

- Frontend: http://localhost:3000
- Backend API: http://localhost:5001
- PostgreSQL: localhost:5432

## Code Style and Conventions

- **Prettier** (`.prettierrc`): semicolons, single quotes, 80 print width, 2 spaces, ES5 trailing commas, LF line endings
- **ESLint** 9 flat config in `backend/eslint.config.js` and `frontend/eslint.config.js`
- **TypeScript** strict mode; both packages use `"type": "module"` (ES modules)
- **Backend**: Register routes in `backend/src/app.ts`; use `errorHandler` as last middleware
- **Frontend**: Use `@/` aliases for `components`, `lib`, `components/ui`, `lib`, `hooks` (see `frontend/components.json`)

## Architectural Patterns

- **Backend**: Express app with CORS, JSON body parser, static uploads at `/uploads`, health check at `/api/health`
- **Frontend**: React Router with `BrowserRouter`, Tailwind for styling, shadcn/ui components
- **Add shadcn components**: `cd frontend && npx shadcn@latest add [component-name]`
- **Migrations**: SQL files in `backend/src/db/migrations/` named `NNN_description.sql`; run via `npm run db:migrate`

## Files/Folders to Be Aware Of

- `backend/env.example` and `frontend/env.example` – copy to `.env` for local config
- `backend/uploads/` – file uploads (gitignored except `.gitkeep`)
- `frontend/components.json` – shadcn/ui config (aliases, style)

## Files/Folders to Avoid

- Do not edit `node_modules/`, `dist/`, `build/`
- Do not commit `.env` files (use `.env.example` as reference)
- `backend/uploads/*` is gitignored
