import os
import cv2
import numpy as np
import onnxruntime as ort

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"P:\FaceTech\Face-Tech\models\arcfaceresnet100-8.onnx"
FACE_IMAGE_PATH = r"P:\FaceTech\Face-Tech\Experiments\Aligned_image.png"

# -----------------------------
# PREPROCESS (CRITICAL)
# -----------------------------
def preprocess_arcface(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))
    
    # Keep BGR, just transpose
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, 0)  # Add batch
    img = img.astype(np.float32)
    
    return img


def l2_normalize(x, eps=1e-10):
    """L2 normalize embeddings to unit length"""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


# -----------------------------
# MAIN
# -----------------------------
def main():
    # Verify paths
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(FACE_IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {FACE_IMAGE_PATH}")
    
    print("✓ Model exists:", MODEL_PATH)
    print("✓ Model size (MB):", round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2))

    # Load ONNX model
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    print("\n✓ MODEL LOADED")
    print(f"  Input name  : {sess.get_inputs()[0].name}")
    print(f"  Input shape : {sess.get_inputs()[0].shape}")
    print(f"  Input dtype : {sess.get_inputs()[0].type}")
    print(f"  Output name : {sess.get_outputs()[0].name}")
    print(f"  Output shape: {sess.get_outputs()[0].shape}")

    # Preprocess face
    print("\n🔄 Preprocessing face...")
    face_tensor = preprocess_arcface(FACE_IMAGE_PATH)

    print(f"  Face tensor shape : {face_tensor.shape}")
    print(f"  Face tensor dtype : {face_tensor.dtype}")
    print(f"  Face tensor range : [{face_tensor.min():.3f}, {face_tensor.max():.3f}]")
    print(f"  Is C-contiguous   : {face_tensor.flags['C_CONTIGUOUS']}")

    # Verify shape matches expected
    expected_shape = sess.get_inputs()[0].shape
    print(f"  Expected shape    : {expected_shape}")

    # Run inference
    print("\n🚀 Running inference...")
    input_name = sess.get_inputs()[0].name
    
    try:
        outputs = sess.run(None, {input_name: face_tensor})
        embedding = outputs[0]
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print(f"\nDebug info:")
        print(f"  Input name: {input_name}")
        print(f"  Input shape: {face_tensor.shape}")
        print(f"  Input dtype: {face_tensor.dtype}")
        raise

    print(f"✓ Raw embedding shape: {embedding.shape}")
    print(f"  Raw embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")

    # Handle embedding shape
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)
    elif len(embedding.shape) == 3:
        # Flatten spatial dimensions if present
        embedding = embedding.reshape(embedding.shape[0], -1)

    # L2 normalize
    embedding_normalized = l2_normalize(embedding)

    print(f"\n✓ Normalized embedding shape: {embedding_normalized.shape}")
    print(f"  Embedding dimension: {embedding_normalized.shape[1]}")
    print(f"  Embedding norm: {np.linalg.norm(embedding_normalized):.6f}")
    print(f"  Self cosine similarity: {float(np.dot(embedding_normalized, embedding_normalized.T)):.6f}")

    # Print first 10 values for inspection
    print(f"\n📊 First 10 embedding values:")
    print(embedding_normalized[0, :10])

    print("\n✅ ARCFACE WORKING SUCCESSFULLY!")
    return embedding_normalized


if __name__ == "__main__":
    try:
        embedding = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()