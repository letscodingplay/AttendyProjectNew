import face_recognition
import cv2
import os

similarity_threshold = 0.4

face_images_directory = "attendy-new/training_data/"
supported_extensions = [".jpg", ".jpeg", ".png"]

# 등록된 얼굴 이미지 로드 및 인코딩
known_face_encodings = {}
known_face_names = []
# 디렉토리 내의 모든 하위 디렉토리(개인별 디렉토리 가져오기)
person_directories = [dir for dir in os.listdir(face_images_directory) if os.path.isdir(os.path.join(face_images_directory, dir))]

# 각 개인별 디렉토리 내의 이미지 파일 로드 및 인코딩
for person_directory in person_directories:
    person_name = person_directory
    
    person_dir_path = os.path.join(face_images_directory, person_directory)
    
    #개인별 디렉토리 내의 이미지 파일 경로 가져오기
    image_paths = [os.path.join(person_dir_path, file) for file in os.listdir(person_dir_path) if os.path.splitext(file)[1].lower() in supported_extensions]
    
    encodings = []
    for image_path in image_paths:    
        face_image = face_recognition.load_image_file(image_path)
        print(f"person_name:{person_name}")
        try:
            face_encoding = face_recognition.face_encodings(face_image)[0]
            encodings.append(face_encoding)
        except IndexError:
            print("Index Out Of Range Pass")        
    
    known_face_encodings[person_name] = encodings

# 웹캠 열기 및 얼굴 인식 진행
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # 등록된 얼굴과 비교하여 가장 유사한 얼굴 찾기
        best_match_name = None
        best_match_distance = float('inf')

        for name, encodings in known_face_encodings.items():
            distances = face_recognition.face_distance(encodings, face_encoding)
            try:
                min_distance = min(distances)
                if min_distance < best_match_distance:
                    best_match_name = name
                    best_match_distance = min_distance
            except :
                print("distance error")
            
        if best_match_distance > similarity_threshold :
            best_match_name = "Unknown"
            
        # 얼굴 위치에 사각형 그리기 및 이름 표시
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, best_match_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Facial Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
video_capture.release()
cv2.destroyAllWindows()