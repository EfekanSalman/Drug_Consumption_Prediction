#!/bin/bash

# Deployment script for drug consumption prediction API

set -e

# Configuration
DOCKER_IMAGE="drug-consumption-prediction"
DOCKER_TAG="latest"
REGISTRY="docker.io"
NAMESPACE="drug-prediction"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    log_info "All dependencies are available"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Push image to registry
push_image() {
    log_info "Pushing image to registry..."
    
    # Tag image for registry
    docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG}
    
    # Push image
    docker push ${REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG}
    
    if [ $? -eq 0 ]; then
        log_info "Image pushed successfully"
    else
        log_error "Failed to push image"
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ -n ${NAMESPACE}
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/drug-consumption-prediction-api -n ${NAMESPACE}
    
    if [ $? -eq 0 ]; then
        log_info "Deployment successful"
    else
        log_error "Deployment failed"
        exit 1
    fi
}

# Deploy with Docker Compose
deploy_compose() {
    log_info "Deploying with Docker Compose..."
    
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        log_info "Docker Compose deployment successful"
    else
        log_error "Docker Compose deployment failed"
        exit 1
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for service to be ready
    sleep 30
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "API health check passed"
    else
        log_warn "API health check failed"
    fi
    
    # Check MLflow
    if curl -f http://localhost:5000 > /dev/null 2>&1; then
        log_info "MLflow health check passed"
    else
        log_warn "MLflow health check failed"
    fi
}

# Main deployment function
main() {
    local deployment_type=${1:-"compose"}
    
    log_info "Starting deployment process..."
    log_info "Deployment type: ${deployment_type}"
    
    check_dependencies
    
    case ${deployment_type} in
        "k8s")
            build_image
            push_image
            deploy_k8s
            ;;
        "compose")
            build_image
            deploy_compose
            health_check
            ;;
        *)
            log_error "Invalid deployment type. Use 'k8s' or 'compose'"
            exit 1
            ;;
    esac
    
    log_info "Deployment completed successfully!"
    
    # Display service URLs
    echo ""
    log_info "Service URLs:"
    echo "  API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  MLflow: http://localhost:5000"
    echo "  Grafana: http://localhost:3000 (admin/admin)"
    echo "  Prometheus: http://localhost:9090"
}

# Run main function with all arguments
main "$@"
