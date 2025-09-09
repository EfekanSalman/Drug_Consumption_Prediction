# Production Deployment Guide

This guide provides comprehensive instructions for deploying the Drug Consumption Prediction API to production environments.

## 🏗️ Architecture Overview

The production system consists of:

- **FastAPI Application**: REST API for model serving
- **MLflow**: Experiment tracking and model registry
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Docker**: Containerization
- **Kubernetes**: Orchestration (optional)
- **GitHub Actions**: CI/CD pipeline

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Git
- kubectl (for Kubernetes deployment)

### Local Development Setup

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd Drug_Consumption_Prediction
   pip install -r requirements-dev.txt
   ```

2. **Train the model:**
   ```bash
   python src/main.py --mode train
   ```

3. **Start the API:**
   ```bash
   python src/api/main.py
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Services Available

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Using the Deployment Script

```bash
# Deploy with Docker Compose
./scripts/deploy.sh compose

# Deploy to Kubernetes
./scripts/deploy.sh k8s
```

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster
- kubectl configured
- Docker registry access

### Deploy to Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n drug-prediction

# Get service URL
kubectl get service -n drug-prediction
```

### Scaling

```bash
# Scale the deployment
kubectl scale deployment drug-consumption-prediction-api --replicas=5 -n drug-prediction
```

## 🔄 CI/CD Pipeline

### GitHub Actions Setup

1. **Set up secrets in GitHub:**
   - `DOCKER_USERNAME`: Docker Hub username
   - `DOCKER_PASSWORD`: Docker Hub password

2. **Pipeline stages:**
   - **Test**: Run unit tests and linting
   - **Train**: Train and validate model
   - **Build**: Build Docker image
   - **Deploy**: Deploy to staging/production

### Manual Triggers

```bash
# Trigger deployment manually
git push origin main
```

## 📊 Monitoring and Observability

### Metrics Collection

The system collects:

- **API Metrics**: Request count, latency, error rate
- **Model Metrics**: Prediction accuracy, confidence scores
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Prediction distribution, drift detection

### Grafana Dashboards

Access Grafana at http://localhost:3000:

- **API Performance**: Request rates, response times
- **Model Performance**: Prediction accuracy, confidence
- **System Health**: Resource utilization
- **Data Drift**: Feature distribution changes

### Alerting

Configure alerts for:

- High error rates (>5%)
- High latency (>1s)
- Data drift detection
- Model performance degradation
- System resource exhaustion

## 🔒 Security Considerations

### API Security

- **Input Validation**: All inputs validated with Pydantic
- **Rate Limiting**: Implement rate limiting for production
- **Authentication**: Add JWT authentication for production
- **HTTPS**: Use TLS certificates in production

### Container Security

- **Non-root User**: Containers run as non-root user
- **Image Scanning**: Scan images for vulnerabilities
- **Secrets Management**: Use Kubernetes secrets or external vault

### Network Security

- **Network Policies**: Restrict pod-to-pod communication
- **Ingress**: Use proper ingress controllers
- **Firewall**: Configure appropriate firewall rules

## 📈 Performance Optimization

### API Performance

- **Async Processing**: FastAPI with async/await
- **Connection Pooling**: Database connection pooling
- **Caching**: Redis for response caching
- **Load Balancing**: Multiple replicas with load balancer

### Model Performance

- **Model Optimization**: Quantization, pruning
- **Batch Processing**: Process multiple requests together
- **GPU Acceleration**: Use GPU for inference if available

### Infrastructure

- **Auto-scaling**: Horizontal Pod Autoscaler
- **Resource Limits**: Set appropriate CPU/memory limits
- **Monitoring**: Continuous performance monitoring

## 🔧 Configuration Management

### Environment Variables

```bash
# API Configuration
MODEL_PATH=/app/models/cannabis_model.pkl
LOG_LEVEL=INFO
PYTHONPATH=/app

# MLflow Configuration
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# Monitoring Configuration
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_ENDPOINT=http://grafana:3000
```

### Configuration Files

- `docker-compose.yml`: Local development
- `k8s/deployment.yaml`: Kubernetes deployment
- `monitoring/prometheus.yml`: Prometheus configuration
- `monitoring/grafana/`: Grafana dashboards and datasources

## 🧪 Testing in Production

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# MLflow health
curl http://localhost:5000/health

# Prometheus health
curl http://localhost:9090/-/healthy
```

### Load Testing

```bash
# Install artillery
npm install -g artillery

# Run load test
artillery run load-test.yml
```

### Integration Testing

```bash
# Run integration tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Logging and Debugging

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about operations
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations

### Log Aggregation

- **Centralized Logging**: Use ELK stack or similar
- **Structured Logging**: JSON format for easy parsing
- **Log Rotation**: Prevent disk space issues

### Debugging

```bash
# View API logs
docker-compose logs -f api

# View all logs
docker-compose logs -f

# Debug container
docker exec -it <container-id> /bin/bash
```

## 🔄 Backup and Recovery

### Data Backup

- **Model Artifacts**: Regular backup of trained models
- **MLflow Data**: Backup experiment data
- **Monitoring Data**: Backup metrics and logs
- **Configuration**: Version control all configs

### Disaster Recovery

- **Multi-region Deployment**: Deploy across regions
- **Automated Failover**: Configure automatic failover
- **Recovery Procedures**: Document recovery steps

## 📊 Cost Optimization

### Resource Optimization

- **Right-sizing**: Use appropriate instance sizes
- **Auto-scaling**: Scale based on demand
- **Spot Instances**: Use spot instances for non-critical workloads

### Monitoring Costs

- **Resource Tracking**: Monitor resource usage
- **Cost Alerts**: Set up cost alerts
- **Regular Review**: Review and optimize regularly

## 🆘 Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check model file path
   - Verify model file exists
   - Check file permissions

2. **API Performance Issues**
   - Check resource limits
   - Monitor memory usage
   - Review query performance

3. **Monitoring Issues**
   - Verify Prometheus configuration
   - Check Grafana datasources
   - Review alert rules

### Support

- **Documentation**: Check this guide and README
- **Logs**: Review application and system logs
- **Issues**: Create GitHub issues for bugs
- **Community**: Join discussions in repository

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

**Note**: This guide assumes basic familiarity with Docker, Kubernetes, and cloud platforms. For production deployments, consider consulting with DevOps engineers and following your organization's security and compliance requirements.
