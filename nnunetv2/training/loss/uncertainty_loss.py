import torch
from torch.nn import Parameter, Module
import pandas as pd


class JointUncertaintyLoss(Module):
    def __init__(self):
        super(JointUncertaintyLoss, self).__init__()
        
        # Uncertainty parameters - log variance for each task
        # [segmentation_task, classification_task]
        self.log_vars = Parameter(torch.zeros(2, requires_grad=True, dtype=torch.float32).cuda())
        
        # Storage for individual losses
        self.seg_loss = None
        self.cls_loss = None
        self.seg_loss_var = None
        self.cls_loss_var = None
        self.log_sigma_reg = None
        self.total_loss = None
        self.total_loss_var = None

    def forward(self, seg_loss, cls_loss):
        """
        Args:
            seg_loss: Pre-computed segmentation loss (dice + cross_entropy)
            cls_loss: Pre-computed classification loss (bce or focal_bce)
        """
        
        # Store the input losses
        self.seg_loss = seg_loss
        self.cls_loss = cls_loss
        
        # Apply uncertainty weighting
        # Using the multi-task uncertainty formulation: L = (1/2σ²)L_task + log(σ)
        # where σ² = exp(log_var), so σ = exp(log_var/2)
        
        # Segmentation loss with uncertainty: (1/2σ²_seg)L_seg
        self.seg_loss_var = 0.5 * torch.exp(-self.log_vars[0]) * self.seg_loss
        
        # Classification loss with uncertainty: (1/2σ²_cls)L_cls  
        self.cls_loss_var = 0.5 * torch.exp(-self.log_vars[1]) * self.cls_loss
        
        # Regularization term: log(σ_seg * σ_cls) = log(σ_seg) + log(σ_cls)
        # log(σ) = log(exp(log_var/2)) = log_var/2
        self.log_sigma_reg = 0.5 * (self.log_vars[0] + self.log_vars[1])
        
        # Total losses
        self.total_loss = self.seg_loss + self.cls_loss
        self.total_loss_var = self.seg_loss_var + self.cls_loss_var + self.log_sigma_reg
        
        return self.total_loss_var
    
    def get_loss(self):
        """Return loss components as DataFrame"""
        d = {
            'total_loss': [self.total_loss.item() if self.total_loss is not None else 0],
            'segmentation_loss': [self.seg_loss.item() if self.seg_loss is not None else 0],
            'classification_loss': [self.cls_loss.item() if self.cls_loss is not None else 0],
            'total_loss_with_uncertainty': [self.total_loss_var.item() if self.total_loss_var is not None else 0],
            'segmentation_loss_with_uncertainty': [self.seg_loss_var.item() if self.seg_loss_var is not None else 0],
            'classification_loss_with_uncertainty': [self.cls_loss_var.item() if self.cls_loss_var is not None else 0]
        }
        return pd.DataFrame(data=d)
    
    def get_uncertainties(self):
        """Return learned uncertainty parameters"""
        variances = torch.exp(self.log_vars.data)
        d = {
            'segmentation_variance': [variances[0].item()],
            'classification_variance': [variances[1].item()],
            'segmentation_log_var': [self.log_vars.data[0].item()],
            'classification_log_var': [self.log_vars.data[1].item()]
        }
        return pd.DataFrame(data=d)
    
    def get_task_weights(self):
        """Return the effective task weights (1/2σ²)"""
        weights = 0.5 * torch.exp(-self.log_vars.data)
        d = {
            'segmentation_weight': [weights[0].item()],
            'classification_weight': [weights[1].item()]
        }
        return pd.DataFrame(data=d)


# Example usage
if __name__ == "__main__":
    # Initialize uncertainty loss
    uncertainty_loss = JointUncertaintyLoss(cuda=False)  # Set to True if using GPU
    
    # Your pre-computed losses from trainer
    seg_loss = torch.tensor(0.5, requires_grad=True)  # Your dice + CE loss
    cls_loss = torch.tensor(0.3, requires_grad=True)  # Your BCE/Focal loss
    
    # Apply uncertainty weighting
    total_loss = uncertainty_loss(seg_loss, cls_loss)
    
    print("Loss components:")
    print(uncertainty_loss.get_loss())
    print("\nUncertainty parameters:")
    print(uncertainty_loss.get_uncertainties())
    print("\nTask weights:")
    print(uncertainty_loss.get_task_weights())