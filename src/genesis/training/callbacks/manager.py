"""
Callback Manager for Native PyTorch Training
=============================================
Orchestrates training callbacks without Composer dependency.
"""

from typing import List, Dict, Any, Optional
import torch


class CallbackManager:
    """
    Manages and orchestrates training callbacks.
    
    Provides hooks for different training events:
    - on_train_start
    - on_epoch_start/end
    - on_batch_start/end  
    - on_validation_start/end
    - on_train_end
    """
    
    def __init__(self, callbacks: List = None):
        """
        Initialize callback manager.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback):
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def on_train_start(self, model: torch.nn.Module, **kwargs):
        """Called at the start of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_start'):
                callback.on_train_start(model, **kwargs)
    
    def on_train_end(self, model: torch.nn.Module, **kwargs):
        """Called at the end of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(model, **kwargs)
    
    def on_epoch_start(self, epoch: int, model: torch.nn.Module, **kwargs):
        """Called at the start of each epoch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(epoch, model, **kwargs)
    
    def on_epoch_end(self, epoch: int, model: torch.nn.Module, metrics: Dict[str, float] = None, **kwargs):
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, model, metrics or {}, **kwargs)
    
    def on_batch_start(self, step: int, model: torch.nn.Module, batch: Any, **kwargs):
        """Called at the start of each batch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_start'):
                callback.on_batch_start(step, model, batch, **kwargs)
    
    def on_batch_end(self, step: int, model: torch.nn.Module, loss: float, **kwargs):
        """Called at the end of each batch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(step, model, loss, **kwargs)
    
    def on_validation_start(self, step: int, model: torch.nn.Module, **kwargs):
        """Called at the start of validation."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_validation_start'):
                callback.on_validation_start(step, model, **kwargs)
    
    def on_validation_end(
        self, 
        step: int, 
        model: torch.nn.Module, 
        val_loss: float,
        metrics: Dict[str, float] = None,
        **kwargs
    ):
        """Called at the end of validation."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_validation_end'):
                callback.on_validation_end(step, model, val_loss, metrics or {}, **kwargs)


class BaseCallback:
    """Base class for callbacks."""
    
    def on_train_start(self, model: torch.nn.Module, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, model: torch.nn.Module, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, epoch: int, model: torch.nn.Module, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, model: torch.nn.Module, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, step: int, model: torch.nn.Module, batch: Any, **kwargs):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, step: int, model: torch.nn.Module, loss: float, **kwargs):
        """Called at the end of each batch."""
        pass
    
    def on_validation_start(self, step: int, model: torch.nn.Module, **kwargs):
        """Called at the start of validation."""
        pass
    
    def on_validation_end(
        self, 
        step: int, 
        model: torch.nn.Module, 
        val_loss: float, 
        metrics: Dict[str, float],
        **kwargs
    ):
        """Called at the end of validation."""
        pass
