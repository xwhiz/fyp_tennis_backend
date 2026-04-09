# TennisProject

## Environment Setup (`.env`)

Create a `.env` file in the project root.  
Use the template below (derived from `src/config.py` and current Docker setup):

```env
# Postgres (docker-compose postgres service)
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=acevision

# Database URL for app/alembic/celery
DATABASE_URL=postgresql://admin:admin123@postgres:5432/acevision

# Application
APP_NAME=AceVision Backend
APP_ENV=development
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=7000

# Celery
CELERY_BROKER_URL=amqp://guest:guest@rabbitmq:5672//
CELERY_RESULT_BACKEND=redis://redis:6379/0
CELERY_APP_NAME=acevision-backend-tasks
CELERY_WORKER_CONCURRENCY=1

# Flower
FLOWER_UNAUTHENTICATED_API=true

# Video processing
VIDEO_BATCH_SIZE=200
UPLOAD_CHUNK_SIZE=20971520

# JWT auth
JWT_SECRET=change-me-with-a-strong-secret-at-least-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRES_IN_HOURS=72

# Admin seeder
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=admin123
ADMIN_FIRST_NAME=Admin
ADMIN_LAST_NAME=User
```

### Which variables are required?

- Required for local Docker run:
  - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
  - `DATABASE_URL`
  - `JWT_SECRET`
  - `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ADMIN_FIRST_NAME`, `ADMIN_LAST_NAME`
- Strongly recommended to keep explicit:
  - `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
  - `APP_ENV`, `LOG_LEVEL`

### Start services

```bash
docker compose up --build
```

After startup:
- API: `http://localhost:7000`
- API docs: `http://localhost:7000/docs`
- Adminer: `http://localhost:8080` (server: `postgres`)
- Flower: `http://localhost:5556`

### Notes

- `docker-compose.yml` now reads DB/admin credentials from `.env` instead of hardcoded values.
- Keep production secrets out of source control.

Number of threads is to be set from the OS environment variable `NUMBER_OF_THREADS`. (NOT WORKING RN)
i.e.
To run the project, you can use the following command:

```bash
NUMBER_OF_THREADS=4 uvicorn main:app --host 0.0.0.0 --port 7000
```

---

## Previous stuff

Tennis analysis using deep learning and machine learning. <br>
You can check this blog post https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee for more details

![](pics/hard.gif)
![](pics/grass.gif)
![](pics/clay.gif)

### Ball detection

TrackNet was used for detecting tennis ball during the game. For more information you can check this repository: https://github.com/yastrebksv/TrackNet. There you can find
pretrained weights to check the model.

### Bounce detection

CatBoostRegressor was used to predict ball's bounces during the game based on ball trajectory detected in the previous step. You can check this pretrained model: https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view?usp=drive_link

### Court detection

It was used neural network for detection 14 points of tennis court. For more information you can check this repository: https://github.com/yastrebksv/TennisCourtDetector. There you can find pretrained weights to check the model.

### How to run

Prepare a video file with resolution 1280x720

1. Clone the repository `https://github.com/yastrebksv/TennisProject.git`
2. Run `pip install -r requirements.txt` to install packages required
3. Run `python main.py <args>`

install radis and rabbitMQ for Celery

for celery dashboard
celery --broker=http://.
