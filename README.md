# Hereon-Segmentation

Hereon-Segmentation is a full-stack image segmentation project including:

- **Backend API** (`backend`) – handles user requests  
- **ML container** (`img-seg`) – runs the image segmentation model  
- **Frontend** (`frontend`) – static website for uploading and segmenting images  

---

## 1. Clone the repository

```bash
git clone https://github.com/louamlemjid/hereon-segementation.git
cd hereon-segementation
````

Check the folder structure:

```bash
ls
# backend  frontend  imageProcessing  node_modules  README.md
```

---

## 2. Pull the Docker images

```bash
docker pull louam/backend:v1.1
docker pull louam/frontend-img:v1.1
docker pull louam/img-seg:torch-v1.0
```

---

## 3. Create a Docker network

```bash
docker network create backendNet
```

This allows the containers to communicate with each other.

---

## 4. Run the backend container

```bash
cd backend

docker run -it \
  --add-host=host.docker.internal:host-gateway \
  -p 8000:8000 \
  -v "$PWD":/app \
  -v /app/node_modules \
  --name backend-container \
  --network backendNet \
  louam/backend:v1.1
```

* `-v "$PWD":/app` mounts the backend code
* `--network backendNet` connects it to other containers

---

## 5. Run the ML image segmentation container

```bash
cd ../imageProcessing

docker run -d \
  --name img-seg-container \
  --network backendNet \
  --gpus all \
  louam/img-seg:torch-v1.0
```

This container serves the ML API for image segmentation.

---

## 6. Run the frontend container

```bash
cd ../frontend

docker run -d -p 8080:80 \
  --name static-site \
  louam/frontend-img:1.1
```

* Frontend will be accessible at [http://localhost:8080](http://localhost:8080)

---

## 7. Test the application

1. Open [http://localhost:8080](http://localhost:8080) in your browser
2. Upload an image
3. Enter a `userId`
4. Click **Segment**
5. The image will be processed by the ML container and the result will appear in the frontend

---

## Optional commands

* **Stop all containers:**

```bash
docker stop backend-container img-seg-container static-site
docker rm backend-container img-seg-container static-site
```

* **View container logs:**

```bash
docker logs -f backend-container
docker logs -f img-seg-container
docker logs -f static-site
```

* **Re-run frontend or backend with volume mounts** for live code updates during development
