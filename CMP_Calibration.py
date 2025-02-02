import torch
import torch.nn as nn
import torch.nn.functional as F

class CMPLoss(nn.Module):
    def __init__(self, epsilon=1e-10):
        super(CMPLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, probs, labels):
        """
        Compute the CMP loss.

        Args:
            probs (torch.Tensor): Predicted probabilities (shape [batch_size, num_classes]).
            labels (torch.Tensor): Ground truth labels (shape [batch_size]).

        Returns:
            torch.Tensor: The CMP loss.
        """
        batch_size, num_classes = probs.shape
        loss = 0.0

        for i in range(batch_size):
            p_xi_yi = probs[i, labels[i]]
            p_xi_yj = probs[i, labels != labels[i]]
            p_xi_yj = p_xi_yj[p_xi_yj > p_xi_yi]

            if len(p_xi_yj) > 0:
                loss += p_xi_yi / (p_xi_yj.sum() + self.epsilon)

        return loss / batch_size

class CLIPWithCMPLoss(nn.Module):
    def __init__(self, clip_model, lambda_cmp=1.0):
        super(CLIPWithCMPLoss, self).__init__()
        self.clip_model = clip_model
        self.cmp_loss = CMPLoss()
        self.lambda_cmp = lambda_cmp

    def forward(self, images, texts, labels):
        """
        Compute the final loss combining CLIP loss and CMP loss.

        Args:
            images (torch.Tensor): Input images (shape [batch_size, ...]).
            texts (torch.Tensor): Input texts (shape [batch_size, ...]).
            labels (torch.Tensor): Ground truth labels (shape [batch_size]).

        Returns:
            torch.Tensor: The combined loss.
        """
        # Compute CLIP loss
        logits_per_image, logits_per_text = self.clip_model(images, texts)
        clip_loss = F.cross_entropy(logits_per_image, labels)

        # Compute CMP loss
        probs = F.softmax(logits_per_image, dim=-1)
        cmp_loss = self.cmp_loss(probs, labels)

        # Combine losses
        final_loss = clip_loss + self.lambda_cmp * cmp_loss

        return final_loss

# Example usage:
# Assuming `clip_model` is your pre-trained CLIP model
# `images`, `texts`, and `labels` are your input data
clip_model = ...  # Your CLIP model
model_with_cmp = CLIPWithCMPLoss(clip_model, lambda_cmp=1.0)

# Training loop
optimizer = torch.optim.Adam(model_with_cmp.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for images, texts, labels in dataloader:
        optimizer.zero_grad()
        loss = model_with_cmp(images, texts, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")