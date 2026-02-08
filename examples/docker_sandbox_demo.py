"""
Docker Sandbox Demo (Bet 5 - Enterprise Security)

Demonstrates isolated code execution using Docker containers for maximum security.
"""

from rlmkit.envs.sandbox import DockerExecutor


def demo_basic_execution():
    """Demo 1: Basic isolated execution."""
    print("=" * 60)
    print("Demo 1: Basic Docker Execution")
    print("=" * 60)
    
    # Check if Docker is available
    if not DockerExecutor.is_available():
        print("⚠️  Docker not available. Install Docker to run this demo.")
        print("   https://docs.docker.com/get-docker/")
        return
    
    print("✅ Docker is available\n")
    
    # Create executor
    executor = DockerExecutor()
    
    # Execute simple code
    code = """
import numpy as np

data = np.array([1, 2, 3, 4, 5])
print(f"Data: {data}")
print(f"Mean: {data.mean()}")
print(f"Sum: {data.sum()}")
"""
    
    print("Executing code in isolated Docker container...")
    result = executor.execute(code)
    
    if result["result"]:
        print("\n✅ Execution successful!")
        print("Output:")
        print(result["output"])
    else:
        print(f"\n❌ Execution failed: {result['error']}")
    print()


def demo_security_isolation():
    """Demo 2: Security features - network isolation."""
    print("=" * 60)
    print("Demo 2: Network Isolation (Security)")
    print("=" * 60)
    
    if not DockerExecutor.is_available():
        print("⚠️  Docker not available")
        return
    
    executor = DockerExecutor(network_mode="none")
    
    # Try to access network (should fail)
    code = """
import socket

try:
    # This should fail due to network isolation
    socket.create_connection(("google.com", 80), timeout=1)
    print("Network access: ALLOWED (unexpected!)")
except Exception as e:
    print(f"Network access: BLOCKED ✅ ({type(e).__name__})")
"""
    
    print("Testing network isolation...")
    result = executor.execute(code)
    
    if result["result"]:
        print("\nOutput:")
        print(result["output"])
    print()


def demo_resource_limits():
    """Demo 3: Resource limits."""
    print("=" * 60)
    print("Demo 3: Resource Limits")
    print("=" * 60)
    
    if not DockerExecutor.is_available():
        print("⚠️  Docker not available")
        return
    
    # Create executor with strict limits
    executor = DockerExecutor(
        memory_limit="256m",  # Only 256MB
        cpu_limit="0.5",  # Half a CPU
        timeout=5  # 5 second timeout
    )
    
    code = """
import sys
import numpy as np

print(f"Python version: {sys.version.split()[0]}")
print("Creating array...")

# This should work within limits
data = np.random.rand(1000, 100)
print(f"Array shape: {data.shape}")
print(f"Memory usage: ~{data.nbytes / 1024 / 1024:.2f} MB")
print("✅ Execution within resource limits")
"""
    
    print("Executing with resource limits (256MB RAM, 0.5 CPU, 5s timeout)...")
    result = executor.execute(code)
    
    if result["result"]:
        print("\nOutput:")
        print(result["output"])
    else:
        print(f"\n❌ Failed: {result['error']}")
    print()


def demo_timeout_protection():
    """Demo 4: Timeout protection."""
    print("=" * 60)
    print("Demo 4: Timeout Protection")
    print("=" * 60)
    
    if not DockerExecutor.is_available():
        print("⚠️  Docker not available")
        return
    
    executor = DockerExecutor(timeout=3)  # 3 second timeout
    
    code = """
import time

print("Starting long computation...")
time.sleep(10)  # This will be killed
print("This won't print")
"""
    
    print("Executing code that exceeds timeout...")
    result = executor.execute(code)
    
    if not result["result"]:
        print(f"\n✅ Timeout protection working: {result['error']}")
    print()


def demo_comparison():
    """Demo 5: Compare security levels."""
    print("=" * 60)
    print("Demo 5: Security Level Comparison")
    print("=" * 60)
    
    print("""
RLMKit provides multiple security levels:

1. **Restricted Builtins** (Default)
   - Limited Python builtins
   - Module import whitelist
   - Good for trusted environments
   - Low overhead

2. **Docker Sandbox** (Bet 5 - NEW!)
   - Full process isolation
   - Resource limits (CPU, memory)
   - Network isolation
   - Timeout protection
   - Best for untrusted code
   - Higher overhead (container startup)

When to use Docker:
✅ Running untrusted user code
✅ Multi-tenant systems
✅ Compliance requirements (isolation)
✅ Resource limiting needed

When restricted builtins are enough:
✅ Internal tools
✅ Performance-critical paths
✅ Trusted code sources
✅ Development/testing
""")
    print()


if __name__ == "__main__":
    print("\n🔒 RLMKit Docker Sandbox Demo\n")
    
    demo_basic_execution()
    demo_security_isolation()
    demo_resource_limits()
    demo_timeout_protection()
    demo_comparison()
    
    print("=" * 60)
    print("Key Features:")
    print("=" * 60)
    print("✓ Process isolation via Docker containers")
    print("✓ Resource limits (CPU, memory, timeout)")
    print("✓ Network isolation (no external access)")
    print("✓ Non-root execution")
    print("✓ Ephemeral containers (destroyed after use)")
    print("=" * 60)
