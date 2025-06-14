"""
Shared pytest fixtures for the visual-pushing-grasping test suite.

This module provides common fixtures that can be used across unit and integration tests.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object for testing."""
    config = Mock()
    config.is_sim = True
    config.obj_mesh_dir = 'objects/blocks'
    config.num_obj = 10
    config.push_rewards = True
    config.experience_replay = True
    config.explore_rate_decay = True
    config.is_testing = False
    config.test_preset_cases = False
    config.save_visualizations = False
    config.load_snapshot = False
    config.snapshot_file = None
    config.random_seed = 1234
    config.force_cpu = False
    config.logging_directory = 'logs'
    return config


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    # Create a 224x224x3 RGB image with random values
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_image():
    """Create a sample depth image for testing."""
    # Create a 224x224 depth image with values in millimeters
    return np.random.uniform(300, 2000, size=(224, 224)).astype(np.float32)


@pytest.fixture
def sample_rgbd_image(sample_rgb_image, sample_depth_image):
    """Create a sample RGB-D image pair for testing."""
    return {
        'rgb': sample_rgb_image,
        'depth': sample_depth_image
    }


@pytest.fixture
def mock_robot():
    """Provide a mock robot object for testing."""
    robot = Mock()
    robot.get_camera_data = MagicMock(return_value=(
        np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
        np.random.uniform(300, 2000, size=(480, 640)).astype(np.float32)
    ))
    robot.grasp = MagicMock(return_value=True)
    robot.push = MagicMock(return_value=True)
    robot.restart_sim = MagicMock()
    robot.check_sim = MagicMock(return_value=True)
    robot.get_obj_positions = MagicMock(return_value=np.random.rand(10, 3))
    robot.get_obj_positions_and_orientations = MagicMock(
        return_value=(np.random.rand(10, 3), np.random.rand(10, 3))
    )
    return robot


@pytest.fixture
def mock_trainer():
    """Provide a mock trainer object for testing."""
    trainer = Mock()
    trainer.forward = MagicMock(return_value=(
        torch.rand(1, 1, 224, 224),  # push predictions
        torch.rand(1, 1, 224, 224),  # grasp predictions
        torch.tensor([0.5])  # state value
    ))
    trainer.get_label_value = MagicMock(return_value=(
        torch.ones(1, 320, 320),  # label
        torch.tensor([1.0])  # label value
    ))
    trainer.backprop = MagicMock()
    trainer.save_model = MagicMock()
    trainer.load_model = MagicMock()
    return trainer


@pytest.fixture
def mock_logger():
    """Provide a mock logger object for testing."""
    logger = Mock()
    logger.save_camera_info = MagicMock()
    logger.save_heightmap = MagicMock()
    logger.save_images = MagicMock()
    logger.save_model = MagicMock()
    logger.save_backup_model = MagicMock()
    logger.save_transitions = MagicMock()
    logger.write_to_log = MagicMock()
    return logger


@pytest.fixture
def sample_heightmap():
    """Create a sample heightmap for testing."""
    return np.random.rand(224, 224).astype(np.float32)


@pytest.fixture
def sample_action():
    """Create a sample action for testing."""
    return {
        'action_type': 'grasp',  # or 'push'
        'position': [112, 112],  # pixel coordinates
        'angle': 45.0,  # rotation angle in degrees
        'heightmap_idx': 0
    }


@pytest.fixture
def vrep_connection():
    """Mock V-REP connection for testing simulation components."""
    connection = Mock()
    connection.simxGetObjectHandle = MagicMock(return_value=(0, 1))
    connection.simxSetObjectPosition = MagicMock(return_value=0)
    connection.simxSetObjectOrientation = MagicMock(return_value=0)
    connection.simxStartSimulation = MagicMock(return_value=0)
    connection.simxStopSimulation = MagicMock(return_value=0)
    connection.simxGetVisionSensorImage = MagicMock(return_value=(
        0,
        [255] * (640 * 480 * 3)  # Flattened RGB image
    ))
    connection.simxGetVisionSensorDepthBuffer = MagicMock(return_value=(
        0,
        [0.5] * (640 * 480)  # Flattened depth buffer
    ))
    return connection


@pytest.fixture
def sample_workspace_limits():
    """Define workspace limits for testing."""
    return np.array([
        [-0.724, -0.276],  # x limits
        [-0.224, 0.224],   # y limits
        [-0.0001, 0.4]     # z limits
    ])


@pytest.fixture
def sample_torch_model():
    """Create a simple torch model for testing."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(64, 1, kernel_size=1)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            return x
    
    return SimpleModel()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return captured_output