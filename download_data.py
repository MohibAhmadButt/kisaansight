import os
import shutil
import kagglehub

# 14 Target Agronomy Classes for Pakistan
TARGET_CLASSES = [
    # Potato (3 classes)
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    
    # Tomato (5 classes)
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    
    # Corn / Maize (4 classes)
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___healthy",
    
    # Pepper / Chili (2 classes)
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy"
]

MAX_TRAIN_SAMPLES_PER_CLASS = 250
MAX_VAL_SAMPLES_PER_CLASS = 50

def copy_sampled_images(src_dir: str, dst_dir: str, max_samples: int):
    os.makedirs(dst_dir, exist_ok=True)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_images = images[:max_samples]
    
    for img_name in selected_images:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    
    return len(selected_images)

def find_dataset_roots(base_path: str):
    """Recursively search for the folders named 'train' and 'valid' or 'val'."""
    train_root = None
    val_root = None

    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            low = d.lower()
            if low == 'train' and train_root is None:
                train_root = os.path.join(root, d)
            elif low in ['valid', 'val'] and val_root is None:
                val_root = os.path.join(root, d)

    return train_root, val_root

def main():
    print("=== KisaanSight Dataset Setup (14 Classes) ===")
    
    # Loads directly from cache without re-downloading
    cache_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print(f"✓ Using cached dataset: {cache_path}\n")

    train_src, val_src = find_dataset_roots(cache_path)
    print(f"Detected Train Source: {train_src}")
    print(f"Detected Val Source:   {val_src}\n")

    if not train_src or not val_src:
        print("❌ Could not automatically locate train/valid directories. Check cache path.")
        return

    dst_train_root = os.path.join("data", "train")
    dst_val_root = os.path.join("data", "val")

    os.makedirs(dst_train_root, exist_ok=True)
    os.makedirs(dst_val_root, exist_ok=True)

    # Build case-insensitive map of source directories
    available_train = {d.lower(): d for d in os.listdir(train_src) if os.path.isdir(os.path.join(train_src, d))}
    available_val = {d.lower(): d for d in os.listdir(val_src) if os.path.isdir(os.path.join(val_src, d))}

    total_train = 0
    total_val = 0

    print("Extracting 14 classes into ./data/ ...\n")
    for cls in TARGET_CLASSES:
        cls_low = cls.lower()
        n_train = 0
        n_val = 0

        # Train Split
        if cls_low in available_train:
            real_folder = available_train[cls_low]
            src_train_cls = os.path.join(train_src, real_folder)
            dst_train_cls = os.path.join(dst_train_root, cls)
            n_train = copy_sampled_images(src_train_cls, dst_train_cls, MAX_TRAIN_SAMPLES_PER_CLASS)
            total_train += n_train
        else:
            print(f"⚠ Train missing for: {cls}")

        # Val Split
        if cls_low in available_val:
            real_folder = available_val[cls_low]
            src_val_cls = os.path.join(val_src, real_folder)
            dst_val_cls = os.path.join(dst_val_root, cls)
            n_val = copy_sampled_images(src_val_cls, dst_val_cls, MAX_VAL_SAMPLES_PER_CLASS)
            total_val += n_val
        else:
            print(f"⚠ Val missing for: {cls}")

        print(f"✓ {cls} -> Train: {n_train} | Val: {n_val}")

    print("\n" + "="*50)
    print("✓ Dataset preparation complete!")
    print(f"Total Training Images:   {total_train}")
    print(f"Total Validation Images: {total_val}")
    print(f"Destination:             {os.path.abspath('data')}")
    print("="*50)

if __name__ == "__main__":
    main()