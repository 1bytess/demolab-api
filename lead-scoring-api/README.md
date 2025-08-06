# Lead Scoring FastAPI Service

A production-ready FastAPI service for lead conversion prediction using machine learning.

## 🏗️ Directory Structure

```
lead-scoring-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker container configuration
├── docker-compose.yml     # Docker Compose for local development
├── start.sh              # Production startup script
├── .dockerignore         # Docker ignore file
├── models/               # Model files directory
│   ├── model.joblib      # Trained ML model (you need to add this)
│   └── preprocessor.joblib # Data preprocessor (you need to add this)
└── README.md            # This file
```

## 📋 Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Your trained model files: `model.joblib` and `preprocessor.joblib`

## 🔧 Setup Instructions

### 1. Place Your Model Files

Create a `models/` directory and place your trained model files:

```bash
mkdir models
# Copy your model files here:
cp /path/to/your/model.joblib models/
cp /path/to/your/preprocessor.joblib models/
```

### 2. Local Development

#### Using Python directly:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python main.py
```

#### Using Docker:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t lead-scoring-api .
docker run -p 8000:8000 lead-scoring-api
```

### 3. Production Deployment

#### With Docker (Recommended):

```bash
# Build the image
docker build -t lead-scoring-api .

# Run in production mode
docker run -d \
  --name lead-scoring-api \
  --restart unless-stopped \
  -p 5000:5000 \
  lead-scoring-api
```

#### With startup script:

```bash
# Make script executable
chmod +x start.sh

# Run in production
./start.sh
```

## 🌐 API Endpoints

### Health Check
- **GET** `/v2/health`
- Returns API status and model loading status

### Root Information
- **GET** `/`
- Returns basic API information

### Lead Prediction
- **POST** `/v2/predict`
- Predicts lead conversion probability

#### Request Example:
```json
{
  "TotalVisits": 5,
  "Page Views Per Visit": 3.2,
  "Total Time Spent on Website": 1850,
  "Lead Origin": "API",
  "Lead Source": "Google",
  "Last Activity": "Email Opened",
  "What is your current occupation": "Working Professional"
}
```

#### Response Example:
```json
{
  "prediction": 1,
  "lead_score": 78.5,
  "label": "Will Convert",
  "timestamp": "2024-06-17T10:30:00",
  "model_version": "1.0.0"
}
```

### Model Information
- **GET** `/v2/models/info`
- Returns information about loaded models

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:5000/v2/docs`
- **ReDoc**: `http://localhost:5000/v2/redoc`

## 🔄 Cloudflare Tunnel Setup

To expose your API through Cloudflare Tunnel:

1. Install cloudflared:
```bash
# Download and install cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

2. Authenticate:
```bash
cloudflared tunnel login
```

3. Create tunnel:
```bash
cloudflared tunnel create lead-scoring-api
```

4. Configure tunnel (create `config.yml`):
```yaml
tunnel: YOUR_TUNNEL_ID
credentials-file: /home/user/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  - hostname: your-domain.com
    service: http://localhost:5000
  - service: http_status:404
```

5. Run tunnel:
```bash
cloudflared tunnel run lead-scoring-api
```

## 🔍 Monitoring

### Health Checks

The API includes built-in health checks:

```bash
# Check API health
curl http://localhost:5000/v2/health

# Check model status
curl http://localhost:5000/v2/models/info
```

### Docker Health Checks

The Docker container includes automatic health checks that monitor:
- API responsiveness
- Model loading status
- Service availability

## 🛠️ Development

### Adding New Features

1. Modify `main.py` to add new endpoints
2. Update `requirements.txt` if new dependencies are needed
3. Rebuild Docker image if using containers

### Testing

```bash
# Test the prediction endpoint
curl -X POST "http://localhost:5000/v2/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TotalVisits": 5,
    "Page Views Per Visit": 3.2,
    "Total Time Spent on Website": 1850,
    "Lead Origin": "API",
    "Lead Source": "Google",
    "Last Activity": "Email Opened",
    "What is your current occupation": "Working Professional"
  }'
```

## 🚨 Troubleshooting

### Common Issues

1. **Models not loading**: Ensure `model.joblib` and `preprocessor.joblib` are in the `models/` directory
2. **Port conflicts**: Change the port in Docker run command or docker-compose.yml
3. **Memory issues**: Increase Docker memory allocation for large models

### Logs

```bash
# View Docker logs
docker logs lead-scoring-api

# Follow logs in real-time
docker logs -f lead-scoring-api
```

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `/app` | Python path |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |

## 🔒 Security Considerations

- The API runs as a non-root user in the container
- CORS is configured (update origins for production)
- Input validation using Pydantic models
- Rate limiting can be added for production use

## 📄 License

This project is part of the Demo Lab portfolio by Ezra H.

## 🤝 Contributing

This is a demo project. For suggestions or improvements, please contact through the main portfolio.