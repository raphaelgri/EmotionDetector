import face_extractor

if __name__ == '__main__':
    extr = face_extractor.FaceExtractor("video.mp4")
    face = extr.extract_faces()
    extr.show_face(face)
