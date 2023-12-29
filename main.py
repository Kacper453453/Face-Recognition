from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import argparse
import keyboard
import cv2


BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

parser = argparse.ArgumentParser(description="Recognize faces")

parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--validate", action="store_true", help="Validate trained model")
parser.add_argument("--video", action="store_true", help="recognize faces in a camera")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")

parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()

Path("train").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("val").mkdir(exist_ok=True)


def encode_known_faces(model="hog", encodings_location: Path = Path(DEFAULT_ENCODINGS_PATH)):
  '''
  Generates facial encodings and saves them along with corresponding names to a file.

  Arguments:
  model -- The face detection model to use (default is "hog")
  encodings_location -- Path to save the generated encodings (default is DEFAULT_ENCODINGS_PATH)
  '''

  names = []
  encodings = []

  # Iterate through all images in the "train" directory
  for filepath in Path("train").glob("*/*"):
    name = filepath.parent.name
    image = face_recognition.load_image_file(filepath)

    # Detect faces in the image
    face_location = face_recognition.face_locations(image, model=model)

    # Generate encodings for the detected faces in the image
    face_encodings = face_recognition.face_encodings(image, face_location)

    # Append the encodings and corresponding names to lists
    for encoding in face_encodings:
      names.append(name)
      encodings.append(encoding)

  # Create a dictionary containing names and encodings
  name_encodings = {"names": names, "encodings": encodings}

  # Print the name encodings (for debugging or monitoring purposes)
  #print(name_encodings)

  # Save the name encodings to a file using pickle
  with encodings_location.open(mode="wb") as f:
    pickle.dump(name_encodings, f)


def recognize_faces(image_location: str, model: str = "hog", encodings_location: Path = Path(DEFAULT_ENCODINGS_PATH)):
  '''
  Detects and recognizes faces in the given image using pre-trained encodings.

  Arguments:
  image_location -- Path to the image to recognize faces in
  model -- The face detection model to use (default is "hog")
  encodings_location -- Path to the pre-trained encodings file (default is DEFAULT_ENCODINGS_PATH)
  '''

  # Load the pre-trained encodings from the pickle file
  with encodings_location.open(mode="rb") as f:
    loaded_encodings = pickle.load(f)

  # Load the input image
  input_image = face_recognition.load_image_file(image_location)

  # Detect faces in the input image and obtain their encodings
  input_face_location = face_recognition.face_locations(input_image, model=model)
  input_face_encodings = face_recognition.face_encodings(input_image, input_face_location)

  # Create a Pillow image object from the input image
  pillow_image = Image.fromarray(input_image)

  # Create an ImageDraw object to draw bounding boxes around detected faces
  draw = ImageDraw.Draw(pillow_image)

  # Detect and recognize faces in the input image
  for bounding_box, unknown_encoding in zip(input_face_location, input_face_encodings):
    name = _recognize_face(unknown_encoding, loaded_encodings)
    if not name:
      name = "Unknown"

    _display_face(draw, bounding_box, name)

  # Display the annotated image with recognized faces
  del draw
  pillow_image.show()

def recognize_face_video(video_location, model:str = "hog",
                         encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    '''
        Captures faces in a video.

        Arguments:
        video_path -- Path to the video file
        model -- The face detection model to use (default is "hog")
        encodings_location -- Path to the pre-trained encodings file (default is DEFAULT_ENCODINGS_PATH)
    '''

    # Load the pre-trained encodings from the pickle file
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    cap = cv2.VideoCapture(video_location)
    face_locations = []

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        # Convert the image from BGR color (which OpenCV uses) to RGB
        # color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,
                                                                255), 2)
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Wait for Enter key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    # cv2.destroyAllWindows()


def _recognize_face(unknown_encoding, loaded_encodings):
  '''
  Identifies each face in the given image.

  Arguments:
  unknown_encoding -- Encodings of the unknown face
  loaded_encodings -- Dictionary containing names and their encodings

  Returns:
  Recognized name of the identified face or "Unknown"
  '''
  boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
  votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)

  if votes:
    return votes.most_common(1)[0][0]
  else:
    return "Unknown"


def _display_face(frame, bounding_box, name):
    '''
    Draw a bounding box on the recognized face and add a caption to that bounding box with the name
    of the identified face, or "Unknown" if it doesn’t match any known face.
    '''

    top, right, bottom, left = bounding_box

    # Draw rectangle around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)  # Adjust color and thickness as needed

    # Add caption with the name
    cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Adjust font size and color as needed

def validate(model: str = "hog"):
  '''
      Validate function to recognize faces in validation images.

      Arguments:
      model -- The face detection model to use (default is "hog")
  '''

  print('**Hold "q" to stop**')
  # Iterate through all files in the "val" directory and its subdirectories
  for filepath in Path("val"). rglob("*"):
    if filepath.is_file():
      recognize_faces(
        image_location=str(filepath.absolute()), model=model
      )

    if keyboard.is_pressed('q'):
        break


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.video:
        recognize_face_video(video_location=0, model=args.m)
