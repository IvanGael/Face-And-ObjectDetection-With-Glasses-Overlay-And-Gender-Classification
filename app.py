import cv2
import numpy as np
import torch
from deepface import DeepFace

class FaceFilter:
    def __init__(self, model_path, glasses_path, conf_threshold=0.9, nms_threshold=0.3, top_k=5000):
        self.face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), conf_threshold, nms_threshold, top_k)
        self.glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        self.glasses = self.scale_image(self.glasses)
        
        # Normalized eye locations in the glasses image
        self.glasses_eye_left = (0.35, 0.4)
        self.glasses_eye_right = (0.65, 0.4)

        # Load YOLOv5 model
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def scale_image(self, image, max_size=640):
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
        return image

    def pad_image(self, image):
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
        glasses_width = int(np.linalg.norm(eye_left - eye_right) * 2.5)
        glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
        glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

        eye_center = (eye_left + eye_right) // 2
        glasses_eye_left = np.array(self.glasses_eye_left) * [glasses_resized.shape[1], glasses_resized.shape[0]]
        glasses_eye_right = np.array(self.glasses_eye_right) * [glasses_resized.shape[1], glasses_resized.shape[0]]
        
        # Calculate the angle to rotate the glasses
        angle = np.degrees(np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))
        
        # Pad the glasses to avoid cropping during rotation
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
        output = image.copy()

        for face in faces:
            landmarks = face[4:14].reshape(5, 2).astype(int)
            eye_left, eye_right = landmarks[0], landmarks[1]

            # Apply glasses to the face using eye landmarks
            output = self.apply_glasses(output, self.glasses, eye_left, eye_right)

            # Extract face region for gender detection
            x, y, w, h = face[:4].astype(int)
            face_region = image[y:y+h, x:x+w]
            if face_region.size > 0:
                try:
                    gender = self.detect_gender(face_region)
                    gender_text = max(gender, key=gender.get)  # Get the gender with the highest probability
                    cv2.putText(output, gender_text, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error detecting gender: {str(e)}")

        return output

    def detect_gender(self, face_image):
        # Use DeepFace to analyze the face and detect gender
        result = DeepFace.analyze(face_image, actions=['gender'], enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            return result[0]['gender']
        elif isinstance(result, dict):
            return result['gender']
        else:
            return {"Unknown": 1.0}

    def visualize_objects(self, image, results):
        # Draw bounding boxes and labels 
        results.render()
        rendered = np.array(results.ims[0])
        return rendered

    def run_face_detection(self, frame):
        self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.face_detector.detect(frame)
        return faces

    def run_object_detection(self, frame):
        # Perform object detection
        results = self.yolo_model(frame)
        return results

def main():
    model_path = 'face_detection_yunet_2023mar.onnx'
    glasses_path = 'glasses.png'
    face_filter = FaceFilter(model_path, glasses_path)

    cap = cv2.VideoCapture("video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

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

        # Write the frame to the output video file
        out.write(frame)

        cv2.imshow('Face and Object Detection with Glasses Overlay', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()