# 🎬 CinemaSense – Emotion-Based Movie Feedback System

CinemaSense is a real-time emotion detection system that captures audience reactions as they exit a movie theatre.
The goal is to get genuine feedback without interrupting the viewing experience — using facial emotion recognition instead of surveys or forms.

This project was developed as part of our Minor Project at St. Joseph’s College of Engineering and Technology (SJCET).

## 🧠 About the Project

- Traditional movie feedback methods like rating forms or interviews are often biased and time-consuming.
- CinemaSense solves this by automatically detecting emotions from faces in a live camera feed.
- When people walk out of the theatre, the system analyzes their expressions and determines what emotions they’re showing — happy, sad, angry, surprised, etc.

It also compares these detected emotions with an expected emotion (for example, “happy” for a comedy movie) and calculates a Reflection Score that shows how well the audience’s reactions matched the tone of the movie.

## ⚙️ Tech Stack

- Python
- OpenCV – for face detection and live video feed
- DeepFace – for emotion recognition
- Tkinter – for the graphical user interface
- Pillow (PIL) – for image display in GUI
- Matplotlib – for charts and analytics
- NumPy – for numerical operations

## ✨ Key Features

- Detects multiple faces in real time using webcam

- Identifies 7 basic emotions: happy, sad, angry, surprise, fear, disgust, neutral

- Calculates a Reflection Score based on expected emotion

- Displays real-time emotion charts and summaries

- Simple GUI for starting and stopping analysis

- Works offline and runs smoothly on normal hardware

## 🏃 How to Run

1. Clone this repository
```bash
git clone https://github.com/<your-username>/CinemaSense.git
cd CinemaSense
```
2. Install the required libraries
```bash
pip install -r requirements.txt
```
3. Run the application
```bash
   python cinemasense_app.py
 ```

4. Enter the movie title and expected emotion, then click Start.
The webcam will open and start detecting emotions in real time.

 ## 🖼️ Output

Live webcam feed showing faces with emotion labels

Reflection score updated in real time

Graphs showing emotion distribution and reflection trends after each session

![CinemaSense GUI](https://github.com/Diyab2003/CinemaSense/blob/main/real-time.png?raw=true)
![Emotion Report](https://github.com/Diyab2003/CinemaSense/blob/main/Summary.png?raw=true)
![Graph](https://github.com/Diyab2003/CinemaSense/blob/main/Graph.png?raw=true)

## 💡 Future Improvements

- Use more advanced face detection models (like MTCNN or RetinaFace)

- Add voice tone or sentiment analysis

- Optimize for better speed with GPU

- Build a web dashboard for multi-theatre usage



