import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

class FaceFilter:
    def __init__(self, model_path, glasses_path, conf_threshold=0.9, nms_threshold=0.3, top_k=5000):
        """
        Initialize the FaceFilter class with the face detection model and glasses image
        Args:
            model_path (str): Path to the face detection model
            glasses_path (str): Path to the glasses image
            conf_threshold (float): Confidence threshold for face detection
            nms_threshold (float): Non-maximum suppression threshold for face detection
            top_k (int): Maximum number of detections to keep
        """
        self.face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), conf_threshold, nms_threshold, top_k)
        self.glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        self.glasses = self.scale_image(self.glasses)
        
        # Normalized eye locations in the glasses image
        self.glasses_eye_left = (0.35, 0.4)
        self.glasses_eye_right = (0.65, 0.4)

        # Load YOLOv8 model
        self.yolo_model = YOLO('yolo11n.pt')  # Use 'n' for nano, can also use 's', 'm', 'l', or 'x' versions

    def scale_image(self, image, max_size=640):
        """
        Scale the image to a maximum size while maintaining the aspect ratio
        Args:
            image (numpy.ndarray): Input image
            max_size (int): Maximum size for the larger dimension
        """
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
        return image

    def pad_image(self, image):
        """
        Pad the image to make it square
        Args:
            image (numpy.ndarray): Input image
        """
        height, width = image.shape[:2]
        max_side = max(height, width)
        padded_image = cv2.copyMakeBorder(image, 
                                          top=(max_side - height) // 2, 
                                          bottom=(max_side - height + 1) // 2, 
                                          left=(max_side - width) // 2, 
                                          right=(max_side - width + 1) // 2, 
                                          borderType=cv2.BORDER_CONSTANT, 
                                          value=[0, 0, 0, 0])
        return padded_image

    def apply_glasses(self, image, glasses, eye_left, eye_right):
        """
        Apply glasses to the face in the image
        Args:
            image (numpy.ndarray): Input image
            glasses (numpy.ndarray): Glasses image
            eye_left (tuple): Left eye location (x, y)
            eye_right (tuple): Right eye location (x, y)
        """
        glasses_width = int(np.linalg.norm(eye_left - eye_right) * 2.5)
        glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
        glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

        eye_center = (eye_left + eye_right) // 2
        glasses_eye_left = np.array(self.glasses_eye_left) * [glasses_resized.shape[1], glasses_resized.shape[0]]
        glasses_eye_right = np.array(self.glasses_eye_right) * [glasses_resized.shape[1], glasses_resized.shape[0]]
        
        angle = np.degrees(np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))
        
        glasses_padded = self.pad_image(glasses_resized)
        glasses_center = (glasses_padded.shape[1] // 2, glasses_padded.shape[0] // 2)
        M = cv2.getRotationMatrix2D(glasses_center, -angle, 1)
        glasses_rotated = cv2.warpAffine(glasses_padded, M, (glasses_padded.shape[1], glasses_padded.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        glasses_eye_center = (glasses_eye_left + glasses_eye_right) // 2
        top_left_x = int(eye_center[0] - glasses_eye_center[0])
        top_left_y = int(eye_center[1] - glasses_eye_center[1])

        alpha_glasses = glasses_rotated[:, :, 3] / 255.0
        for i in range(glasses_rotated.shape[0]):
            for j in range(glasses_rotated.shape[1]):
                y_offset = int(top_left_y + i - glasses_padded.shape[0] // 2 + glasses_eye_center[1])
                x_offset = int(top_left_x + j - glasses_padded.shape[1] // 2 + glasses_eye_center[0])
                if 0 <= y_offset < image.shape[0] and 0 <= x_offset < image.shape[1]:
                    image[y_offset, x_offset] = image[y_offset, x_offset] * (1 - alpha_glasses[i, j]) + glasses_rotated[i, j, :3] * alpha_glasses[i, j]
        return image

    def visualize_faces(self, image, faces):
        """
        Visualize the faces in the image
        Args:
            image (numpy.ndarray): Input image
            faces (numpy.ndarray): Detected faces
        """
        output = image.copy()

        for face in faces:
            landmarks = face[4:14].reshape(5, 2).astype(int)
            eye_left, eye_right = landmarks[0], landmarks[1]

            output = self.apply_glasses(output, self.glasses, eye_left, eye_right)

            x, y, w, h = face[:4].astype(int)
            face_region = image[y:y+h, x:x+w]
            if face_region.size > 0:
                try:
                    gender = self.detect_gender(face_region)
                    gender_text = max(gender, key=gender.get)
                    cv2.putText(output, gender_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error detecting gender: {str(e)}")

        return output

    def detect_gender(self, face_image):
        """
        Detect gender from the face image
        Args:
            face_image (numpy.ndarray): Input face image
        """
        result = DeepFace.analyze(face_image, actions=['gender'], enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            return result[0]['gender']
        elif isinstance(result, dict):
            return result['gender']
        else:
            return {"Unknown": 1.0}

    def visualize_objects(self, results):
        """
        Visualize the objects detected in the image
        Args:
            results (list): List of detection results
        """
        annotated_frame = results[0].plot()
        return annotated_frame

    def run_face_detection(self, frame):
        """
        Run face detection on the input frame
        Args:
            frame (numpy.ndarray): Input frame
        """
        self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.face_detector.detect(frame)
        return faces

    def run_object_detection(self, frame):
        """
        Run object detection on the input frame
        Args:
            frame (numpy.ndarray): Input frame
        """
        results = self.yolo_model(frame)
        return results

def main():
    model_path = 'face_detection_yunet_2023mar.onnx'
    glasses_path = 'glasses.png'
    face_filter = FaceFilter(model_path, glasses_path)

    cap = cv2.VideoCapture("video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output2.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_filter.run_face_detection(frame)
        if faces is not None:
            frame = face_filter.visualize_faces(frame, faces)

        object_detection_results = face_filter.run_object_detection(frame)
        if object_detection_results is not None:
            frame = face_filter.visualize_objects(frame, object_detection_results)

        out.write(frame)
        cv2.imshow('Face and Object Detection with Glasses Overlay', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()