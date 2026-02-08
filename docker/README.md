# RLMKit Docker Sandbox

Docker-based isolated execution environment for RLM code execution.

## Quick Start

### Build the Image

```bash
cd docker
docker build -t rlmkit-sandbox -f Dockerfile.sandbox .
```

### Use with RLMKit

```python
from rlmkit.envs.sandbox import DockerExecutor

# Check if Docker is available
if DockerExecutor.is_available():
    executor = DockerExecutor()
    
    # Execute code in isolated container
    result = executor.execute("""
import numpy as np
data = np.array([1, 2, 3, 4, 5])
print(f"Mean: {data.mean()}")
""")
    
    print(result["output"])
else:
    print("Docker not available - falling back to restricted execution")
```

## Security Features

- ✅ **Process Isolation**: Each execution in separate container
- ✅ **Resource Limits**: Memory (512MB) and CPU (1 core) limits
- ✅ **Network Isolation**: No external network access (`--network=none`)
- ✅ **Non-Root User**: Runs as unprivileged user (uid 1000)
- ✅ **Ephemeral**: Containers destroyed after execution (`--rm`)
- ✅ **Read-Only Code**: Script mounted read-only
- ✅ **Timeout Protection**: 30-second default timeout

## Configuration

```python
executor = DockerExecutor(
    image_name="rlmkit-sandbox",
    memory_limit="512m",  # Adjust as needed
    cpu_limit="1",  # Number of CPUs
    timeout=30,  # Seconds
    network_mode="none"  # "none" or "bridge"
)
```

## Enterprise Deployment

For production use:

1. Pre-build and distribute the Docker image
2. Use container registries (Docker Hub, AWS ECR, etc.)
3. Implement additional monitoring and logging
4. Consider Kubernetes for orchestration

## Troubleshooting

### Docker not found
- Install Docker: https://docs.docker.com/get-docker/
- Ensure Docker daemon is running: `docker ps`

### Permission denied
- Add user to docker group: `sudo usermod -aG docker $USER`
- Restart shell or log out/in

### Image build fails
- Check Dockerfile syntax
- Ensure internet connection for package downloads
- Try: `docker build --no-cache ...`
