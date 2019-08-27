from face_detector import FaceDetector
import base64

detector = FaceDetector("models/faceboxes/saved_model.pb", gpu_memory_fraction=1, visible_device_list='0')

detector.export()

