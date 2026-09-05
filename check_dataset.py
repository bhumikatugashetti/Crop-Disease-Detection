import os

def check_dataset(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"Directory '{dataset_dir}' does not exist. Please extract the dataset here.")
        return

    # The crops we are interested in for this version
    target_crops = ["Tomato", "Potato", "Corn", "Pepper", "Apple"]
    
    try:
        classes = os.listdir(dataset_dir)
    except FileNotFoundError:
        print(f"Error: Cannot read directory {dataset_dir}")
        return

    # Filter directories that match our target crops
    relevant_classes = []
    for c in classes:
        c_path = os.path.join(dataset_dir, c)
        if os.path.isdir(c_path):
            # PlantVillage class names usually look like 'Tomato___Bacterial_spot'
            if any(crop.lower() in c.lower() for crop in target_crops):
                relevant_classes.append(c)

    if not relevant_classes:
        print(f"No relevant crop classes found in the '{dataset_dir}' directory.")
        print("Make sure you extracted the folders (e.g., 'Tomato___healthy', 'Potato___Early_blight') directly inside the 'dataset' folder.")
        return

    print("--- Dataset Statistics ---")
    print(f"Total relevant classes found: {len(relevant_classes)}\n")

    total_images = 0
    for c in sorted(relevant_classes):
        c_path = os.path.join(dataset_dir, c)
        # Count only image files
        images = [f for f in os.listdir(c_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)
        total_images += num_images
        print(f"Class: {c.ljust(35)} | Images: {num_images}")

    print(f"\nTotal number of relevant images: {total_images}")

if __name__ == "__main__":
    # Assuming this script is run from the project root
    dataset_path = os.path.join(os.getcwd(), "dataset")
    check_dataset(dataset_path)
