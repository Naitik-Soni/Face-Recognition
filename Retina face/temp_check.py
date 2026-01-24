import insightface
import onnxruntime as ort
app = insightface.app.FaceAnalysis()
print('GPU Providers:', ort.get_available_providers())
print('✅ InsightFace installed correctly!')