# LightRAG based on HepAI and Open Embedding Model

## Quick Start

### 1.Install Requirements

```
pip install -r requirements.txt
```

### 2.Setup environment variables

```env
# .env file in lightrag_hepai/
Embedding_model_path="/path/to/embedding_model"
embedding_dimension=your_embedding_dimension
max_token_size=your_max_token_size
gpu_id=your_gpu_id
llm_api_key="your_api_key"
llm_base_url="your_base_url"
llm_model_name="openai/gpt-4o-mini"
working_dir="/path/to/working_dir"
```

### 3.Run LightRAG

```
python run_worker.py
```

### 4.Run LightRAG with Docker

#### 4.1.Build the Docker Image

Run the following command in the root directory to build the Docker image:

```bash
docker build -t lightrag-backend:latest .
```

#### 4.2.Run the Docker Container

Start a new container from the image:

```bash
docker run -d -p 4260:4260 --name lightrag-backend-container lightrag-backend:latest
```

####  4.3.Verify the Deployment

Check if the container is running:

```bash
docker ps
```

You should see an active container named `lightrag-backend-container`.

####  4.4.Access the Application

Open your browser and navigate to [http://localhost:4260](http://localhost:4260) to access the FastAPI backend.

- **Swagger UI:** [http://localhost:4260/docs](http://localhost:4260/docs)

### 5.Managing the Docker Container

#### 5.1.Stop the Container

```bash
docker stop lightrag-backend-container
```

#### 5.2.Start the Container Again

```bash
docker start lightrag-backend-container
```

#### 5.3.Remove the Container

First, stop the container if it's running:

```bash
docker stop lightrag-backend-container
```

Then, remove it:

```bash
docker rm lightrag-backend-container
```

#### 5.4.Remove the Docker Image

```bash
docker rmi lightrag-backend:latest
```

#### 5.5.Additional Tips

- **Logs:** To view logs from the running container, use:

  ```bash
  docker logs -f lightrag-backend-container
  ```

- **Interactive Shell:** To access the container's shell:

  ```bash
  docker exec -it lightrag-backend-container /bin/bash
  ```

- **Rebuilding the Image:** If you make changes to the code or dependencies, rebuild the image:

  ```bash
  docker build -t lightrag-backend:latest .
  docker stop lightrag-backend-container
  docker rm lightrag-backend-container
  docker run -d -p 4260:4260 --name lightrag-backend-container lightrag-backend:latest
  ```

### 6.Testing LightRAG

```
python request_lightrag.py
```

