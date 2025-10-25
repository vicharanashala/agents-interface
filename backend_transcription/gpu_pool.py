"""
GPU/MIG Pool Manager for Transcription Service
Manages GPU resources and assigns requests to available MIG slices.
"""

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch

logger = logging.getLogger(__name__)

class GPUStatus(Enum):
    """GPU slice status"""
    FREE = "free"
    BUSY = "busy"
    ERROR = "error"

@dataclass
class GPUSlice:
    """Represents a GPU/MIG slice"""
    device_id: str
    device_name: str
    status: GPUStatus
    assigned_request: Optional[str] = None
    last_used: float = 0.0
    error_count: int = 0

class GPUPool:
    """Manages GPU/MIG pool for transcription requests"""
    
    def __init__(self, mig_devices: List[str]):
        """
        Initialize GPU pool with MIG devices.
        
        Args:
            mig_devices: List of MIG device IDs (e.g., ['MIG-587837bd-78f7-5d96-ad3b-568d0b1febb9', ...])
        """
        self.mig_devices = mig_devices
        self.slices: Dict[str, GPUSlice] = {}
        self.lock = threading.Lock()
        self.request_queue = asyncio.Queue()
        
        # Initialize GPU slices
        self._initialize_slices()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info(f"GPU Pool initialized with {len(self.slices)} MIG devices")
    
    def _initialize_slices(self):
        """Initialize GPU slices from MIG devices"""
        for i, device_id in enumerate(self.mig_devices):
            slice_id = f"mig_{i}"
            self.slices[slice_id] = GPUSlice(
                device_id=device_id,
                device_name=f"MIG-{i}",
                status=GPUStatus.FREE
            )
            logger.info(f"Initialized GPU slice {slice_id}: {device_id}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(30)  # Check every 30 seconds
                    self._cleanup_stale_assignments()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_stale_assignments(self):
        """Clean up stale assignments (older than 5 minutes)"""
        current_time = time.time()
        stale_threshold = 300  # 5 minutes
        
        with self.lock:
            for slice_id, slice_info in self.slices.items():
                if (slice_info.status == GPUStatus.BUSY and 
                    current_time - slice_info.last_used > stale_threshold):
                    logger.warning(f"Cleaning up stale assignment on {slice_id}")
                    slice_info.status = GPUStatus.FREE
                    slice_info.assigned_request = None
                    slice_info.error_count += 1
    
    async def get_free_slice(self, request_id: str, timeout: int = 60) -> Optional[str]:
        """
        Get a free GPU slice for a request.
        
        Args:
            request_id: Unique identifier for the request
            timeout: Maximum time to wait for a free slice (seconds)
            
        Returns:
            GPU slice ID if available, None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                # Find a free slice - prioritize least recently used
                free_slices = [(slice_id, slice_info) for slice_id, slice_info in self.slices.items() 
                              if slice_info.status == GPUStatus.FREE]
                
                if free_slices:
                    # Sort by last_used to get least recently used slice
                    free_slices.sort(key=lambda x: x[1].last_used)
                    slice_id, slice_info = free_slices[0]
                    
                    # Assign the slice
                    slice_info.status = GPUStatus.BUSY
                    slice_info.assigned_request = request_id
                    slice_info.last_used = time.time()
                    
                    logger.info(f"Assigned GPU slice {slice_id} to request {request_id}")
                    return slice_id
            
            # No free slice available, wait a bit
            await asyncio.sleep(0.01)  # Reduced from 0.1s to 0.01s for faster response
        
        logger.warning(f"No free GPU slice available for request {request_id} within {timeout}s")
        return None
    
    def release_slice(self, slice_id: str, request_id: str):
        """
        Release a GPU slice after processing.
        
        Args:
            slice_id: GPU slice ID to release
            request_id: Request ID that was using the slice
        """
        with self.lock:
            if slice_id in self.slices:
                slice_info = self.slices[slice_id]
                if slice_info.assigned_request == request_id:
                    slice_info.status = GPUStatus.FREE
                    slice_info.assigned_request = None
                    slice_info.last_used = time.time()
                    logger.info(f"Released GPU slice {slice_id} from request {request_id}")
                else:
                    logger.warning(f"Attempted to release slice {slice_id} with wrong request ID")
            else:
                logger.error(f"Attempted to release unknown slice {slice_id}")
    
    def mark_slice_error(self, slice_id: str, request_id: str):
        """
        Mark a GPU slice as having an error.
        
        Args:
            slice_id: GPU slice ID that had an error
            request_id: Request ID that encountered the error
        """
        with self.lock:
            if slice_id in self.slices:
                slice_info = self.slices[slice_id]
                slice_info.status = GPUStatus.ERROR
                slice_info.error_count += 1
                slice_info.assigned_request = None
                logger.error(f"Marked GPU slice {slice_id} as error for request {request_id}")
    
    def get_slice_info(self, slice_id: str) -> Optional[GPUSlice]:
        """Get information about a specific GPU slice"""
        with self.lock:
            return self.slices.get(slice_id)
    
    def get_pool_status(self) -> Dict:
        """Get current status of all GPU slices"""
        with self.lock:
            status = {
                "total_slices": len(self.slices),
                "free_slices": sum(1 for s in self.slices.values() if s.status == GPUStatus.FREE),
                "busy_slices": sum(1 for s in self.slices.values() if s.status == GPUStatus.BUSY),
                "error_slices": sum(1 for s in self.slices.values() if s.status == GPUStatus.ERROR),
                "slices": {}
            }
            
            for slice_id, slice_info in self.slices.items():
                status["slices"][slice_id] = {
                    "device_id": slice_info.device_id,
                    "device_name": slice_info.device_name,
                    "status": slice_info.status.value,
                    "assigned_request": slice_info.assigned_request,
                    "last_used": slice_info.last_used,
                    "error_count": slice_info.error_count
                }
            
            return status
    
    def get_device_for_slice(self, slice_id: str) -> str:
        """Get the CUDA device string for a GPU slice"""
        if slice_id in self.slices:
            # All MIG devices are on GPU 1, so always return cuda:1
            return "cuda:1"
        raise ValueError(f"Unknown slice ID: {slice_id}")
    

# Global GPU pool instance
gpu_pool: Optional[GPUPool] = None

def initialize_gpu_pool(mig_devices: List[str]) -> GPUPool:
    """Initialize the global GPU pool"""
    global gpu_pool
    gpu_pool = GPUPool(mig_devices)
    return gpu_pool

def get_gpu_pool() -> Optional[GPUPool]:
    """Get the global GPU pool instance"""
    return gpu_pool
