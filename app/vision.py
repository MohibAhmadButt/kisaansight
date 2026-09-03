import os
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import numpy as np
import cv2

DISEASE_CLASSES = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy'
]

class CropVisionModel:
    def __init__(self, model_path: str = "models/mobilenet_plant.pth"):
        self.device = torch.device("cpu")
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = torch.nn.Linear(in_features, len(DISEASE_CLASSES))
        
        target_path = model_path if os.path.exists(model_path) else "mobilenet_plant.pth"
        if os.path.exists(target_path):
            self.model.load_state_dict(torch.load(target_path, map_location=self.device))
            print(f"✓ Loaded trained weights from {target_path}")
        else:
            print("⚠ Trained weights not found. Using baseline initialization.")
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.features[-1]
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def get_foliage_mask(self, image_pil: Image.Image) -> np.ndarray:
        """Generates boolean foliage mask on RGB color distribution."""
        img_np = np.array(image_pil.convert("RGB"))
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        return (g > 35) & (g > r * 0.80) & (g > b * 0.88)

    def validate_foliage(self, image_pil: Image.Image) -> bool:
        mask = self.get_foliage_mask(image_pil)
        total_pixels = mask.size
        foliage_pixels = np.sum(mask)
        return (foliage_pixels / total_pixels) > 0.06

    def generate_gradcam_and_severity(self, input_tensor: torch.Tensor, class_idx: int, original_pil: Image.Image):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target_score = output[0, class_idx]
        target_score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).squeeze().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        orig_np = np.array(original_pil.convert("RGB"))
        heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
        active_lesion_mask = (heatmap_resized > 0.60)

        # FOLIAGE-RELATIVE SEVERITY MASKING (Constraint to Leaf Silhouette)
        foliage_mask = self.get_foliage_mask(original_pil)
        foliage_pixel_count = max(int(np.sum(foliage_mask)), 1)
        
        # Only count lesions that fall INSIDE the actual leaf
        lesions_on_leaf = active_lesion_mask & foliage_mask
        severity_percent = round(float(np.sum(lesions_on_leaf) / foliage_pixel_count * 100), 1)

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)
        
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        return buffer.tobytes(), severity_percent

    def predict(self, image_bytes: bytes, crop_hint: str = "Auto") -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if not self.validate_foliage(image):
            return {
                "valid_leaf": False,
                "disease": "Invalid_Leaf_Sample",
                "crop": "Unknown",
                "confidence": 0.0,
                "second_candidate": None,
                "second_confidence": 0.0,
                "margin": 0.0,
                "severity_score": 0.0,
                "heatmap_bytes": None
            }

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad = True
        outputs = self.model(tensor)
        
        # Temperature-calibrated softmax (T=1.3)
        probs = torch.softmax(outputs / 1.3, dim=1)[0].clone()

        # Crop Prior Masking
        if crop_hint != "Auto":
            mask = torch.zeros_like(probs)
            for idx, cname in enumerate(DISEASE_CLASSES):
                if crop_hint.lower() in cname.lower():
                    mask[idx] = 1.0
            if mask.sum() > 0:
                probs = probs * mask
                probs = probs / probs.sum()

        top2_prob, top2_idx = torch.topk(probs, 2)
        top1_idx = int(top2_idx[0].item())
        confidence = float(top2_prob[0].item())

        # Out-of-distribution rejection cutoff (< 35%)
        if confidence < 0.35:
            return {
                "valid_leaf": True,
                "disease": "Unknown_Pathogen_Out_Of_Distribution",
                "crop": crop_hint if crop_hint != "Auto" else "Crop",
                "confidence": round(confidence, 4),
                "second_candidate": None,
                "second_confidence": 0.0,
                "margin": 0.0,
                "severity_score": 0.0,
                "heatmap_bytes": None
            }

        top2_idx_val = int(top2_idx[1].item())
        second_conf = float(top2_prob[1].item())
        margin = confidence - second_conf

        predicted_disease = DISEASE_CLASSES[top1_idx]
        second_candidate = DISEASE_CLASSES[top2_idx_val]

        heatmap_bytes, severity_score = self.generate_gradcam_and_severity(tensor, top1_idx, image)
        crop = predicted_disease.split("___")[0].replace("Pepper,_bell", "Pepper").replace("Corn_(maize)", "Corn")

        return {
            "valid_leaf": True,
            "disease": predicted_disease,
            "crop": crop,
            "confidence": round(confidence, 4),
            "second_candidate": second_candidate,
            "second_confidence": round(second_conf, 4),
            "margin": round(margin, 4),
            "severity_score": severity_score,
            "heatmap_bytes": heatmap_bytes
        }