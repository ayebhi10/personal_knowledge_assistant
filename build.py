import logging
import subprocess
import sys
from pathlib import Path

# Configure logging to both console and a log file
logger = logging.getLogger("project_setup")
logger.setLevel(logging.INFO)

# Formatter for uniform log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File Handler (stores setup logs locally)
file_handler = logging.FileHandler("setup.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def run_command(command: str, description: str):
    logger.info(f"Starting process: {description}")
    try:
        # Capture outputs to prevent terminal clutter, logging stdout/stderr only on errors
        result = subprocess.run(
            command, 
            check=True, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        logger.info(f"Successfully completed: {description}")
        logger.debug(f"Command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed execution during: {description}")
        logger.error(f"Command executed: {command}")
        logger.error(f"Exit code: {e.returncode}")
        if e.stdout:
            logger.error(f"Standard Output:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Standard Error:\n{e.stderr}")
        sys.exit(1)

def main():
    project_root = Path(__file__).parent.resolve()
    logger.info("Initializing environment setup script.")
    
    # 1. Directory Structure Creation
    logger.info("Verifying and creating project directories...")
    directories = [
        project_root / "data" / "documents",
        project_root / "data" / "vector_db",
        project_root / "src"
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory.relative_to(project_root)}")
        else:
            logger.debug(f"Directory already exists: {directory.relative_to(project_root)}")
    
    # 2. Poetry Environment Configuration
    run_command(
        "poetry config virtualenvs.in-project true",
        "Configuring Poetry to target local .venv directory"
    )
    
    # 3. Dependency Locking and Installation
    run_command(
        "poetry lock",
        "Generating immutable dependency lockfile (poetry.lock)"
    )
    
    run_command(
        "poetry install",
        "Provisioning virtual environment and sync installing dependencies"
    )
    
    logger.info("Environment orchestration finalized successfully.")
    logger.info("To enter your virtual environment context, execute:")
    logger.info("   source .venv/bin/activate  (Linux/macOS) OR .\\.venv\\Scripts\\activate (Windows)")

if __name__ == "__main__":
    main()