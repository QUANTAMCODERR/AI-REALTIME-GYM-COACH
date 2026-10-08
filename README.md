# AI Real-time GYM Coach

AI Real-time GYM Coach is a Streamlit-based computer-vision fitness assistant that monitors a user’s pose in real time, estimates exercise form quality, counts repetitions, and provides coaching feedback through an AI voice pipeline. The application is designed for interactive workout assistance with live camera analysis and voice-guided corrections.

The system combines:
- real-time pose estimation using MediaPipe
- geometric exercise detection using joint-angle heuristics
- structured workout tracking in SQLite
- LLM-based coaching with Groq
- text-to-speech feedback using gTTS

This project is not a generic ML demo; it is a practical runtime coaching system for exercise form detection and interactive athlete guidance.

## 1. High-level overview

The application runs in Streamlit and displays a webcam-driven workout dashboard. Users select a workout type, set target reps/sets, and start a session. The system then streams frames to a custom WebRTC video processor, extracts 33-body-keypoint landmarks via MediaPipe, evaluates the body pose against exercise-specific geometric rules, updates metrics, and feeds form issues into an AI coach that speaks recommendations aloud.

The key operational loop is:
1. Capture webcam frame
2. Detect pose landmarks
3. Compute joint angles and motion metrics
4. Determine exercise state (up/down, depth, alignment, balance, etc.)
5. Compare against thresholds and form rules
6. Convert form issues to concise feedback text
7. Send text to Groq LLM for style-optimized coaching language
8. Generate audio with TTS
9. Play audio in-app and update the UI dashboard

## 2. Core technical stack

### Frontend and app runtime
- Python 3.12
- Streamlit
- streamlit-webrtc
- OpenCV (cv2)
- NumPy

### Computer vision and pose estimation
- MediaPipe Tasks API
- `vision.PoseLandmarker` from `mediapipe.tasks.python.vision`
- `ml_models/pose_landmarker_full.task` model asset

### Exercise logic and tracking
- custom detector classes in `detectors/`
- geometric angle math via reusable base class
- goal-based rep counting and form evaluation

### AI coaching layer
- Groq Python SDK
- LLM model: `openai/gpt-oss-120b`
- custom prompt system for coaching behavior

### Audio output
- gTTS
- browser playback via `st.audio(..., autoplay=True)`

### Persistence
- SQLite database stored at `data.db`
- user account and exercise history storage

## 3. Important ML model library and function used

This project uses the MediaPipe pose estimation pipeline, specifically the MediaPipe Tasks API with a pose landmark detector.

The crucial code path is in `services/vision/exercise_video_processor.py`:

```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.join(os.getcwd(), "ml_models", "pose_landmarker_full.task")
base_option = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_option,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    output_segmentation_masks=False,
)

self._landmarker = vision.PoseLandmarker.create_from_options(options)
```

Then the model is invoked in the WebRTC loop:

```python
result = self._landmarker.detect_for_video(mp_image, self._frame_timestamps_ms)
```

This means the active ML model function is:
- `vision.PoseLandmarker.create_from_options(...)`
- `detect_for_video(...)`

The model asset is:
- `ml_models/pose_landmarker_full.task`

This is a MediaPipe pose-estimation model that returns landmark coordinates and visibility scores for human joints. The project then uses those 2D/3D-like landmark points to calculate angles and classify exercise quality.

It is important to note that this is not a deep learning classifier for exercise type itself. The actual exercise recognition is rule-based, using the landmark geometry and joint-angle thresholds defined in the detectors.

## 4. LLM coaching model used

The assistant is also connected to Groq for natural-language coaching feedback. In `services/coaching/llm.py`:

```python
response = self.client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    temperature=0.4,
)
```

This means the project uses the Groq-hosted OpenAI OSS model `openai/gpt-oss-120b` as the conversational coach. The model does not control exercise detection directly; it receives a short prompt and a form issue description and returns a concise coaching cue that is spoken to the user.

## 5. System architecture

### 5.1 Entry point
The entry point is `main.py`.

Responsibilities:
- configures the Streamlit page
- initializes the local SQLite database
- renders the login wall
- initializes session state defaults
- sets up voice pipeline and LLM/TTS services
- builds the sidebar workout planner
- starts and stops the exercise session
- embeds the video processor and real-time exercise metrics dashboard

### 5.2 Video processing layer
The file `services/vision/exercise_video_processor.py` is the core runtime loop.

Responsibilities:
- initialize `PoseLandmarker`
- maintain the active exercise type
- draw pose skeleton overlays
- compute metrics for the selected exercise
- send metrics into the detection pipeline
- handle no-pose-detected conditions
- render annotated video frames back into Streamlit WebRTC

### 5.3 Detector layer
The `detectors/` package contains exercise-specific detectors:
- `squat.py`
- `pushup.py`
- `biceps_curl.py`
- `shoulder_press.py`
- `lunges.py`

Each detector extends a common base class from `core/base_exercise.py` and calculates joint angles using landmarks such as shoulders, elbows, knees, and wrists.

Example from `detectors/squat.py`:

```python
left_knee_angle = self.calculate_angle(
    self.get_point(landmarks, self.LEFT_HIP),
    self.get_point(landmarks, self.LEFT_KNEE),
    self.get_point(landmarks, self.LEFT_ANKLE),
)
```

This is a classic geometric pose-analysis method: compute vector angles from landmark positions, classify them into thresholds such as `DOWN_THRESHOLD` / `UP_THRESHOLD`, and count reps whenever a movement stage transitions from down to up.

### 5.4 Coaching layer
The coaching system is split across:
- `services/coaching/llm.py` for generation of feedback text
- `services/coaching/voice_pipeline.py` for contextual issue detection and TTS orchestration
- `services/coaching/tts.py` for audio generation

The `VoicePipeline` class maps raw metrics into user-friendly issues such as:
- squat too high
- knees not bending enough
- back leaning forward
- hips sagging in push-ups
- elbow drift in curls
- excessive arch in overhead press
- loss of balance in lunges

Then it calls the LLM `give_feedback` method to produce short motivational form instructions.

### 5.5 Persistence layer
The SQLite database is initialized in `services/persistence/exercise_repository.py`.

Tables:
- `users`
- `exercises`

This allows:
- per-user authentication identity
- historical exercise logging
- daily aggregation of workout metrics such as reps, sets, and elapsed time

## 6. Supported exercises

The app is configured with the following exercise list in `services/config/workout_config.py`:

- Squats
- Push-ups
- Biceps Curls (Dumbbell)
- Shoulder Press
- Lunges

These are evaluated with custom landmark rules and thresholds. For example:
- Squats: knee angle, back angle, depth status
- Push-ups: elbow angle, body alignment, hip position
- Biceps curls: elbow angle, shoulder stability, swing detection
- Shoulder press: elbow angle, arm extension, lower-back arching
- Lunges: front knee angle, torso angle, balance

## 7. Pose estimation and detection details

The project uses MediaPipe’s pose tracker to estimate the positions of key points such as:
- shoulders (11, 12)
- elbows (13, 14)
- wrists (15, 16)
- hips (23, 24)
- knees (25, 26)
- ankles (27, 28)

The pose skeleton is overlaid on the video using `POSE_CONNECTIONS` defined in `services/config/workout_config.py`.

The drawing pipeline uses OpenCV to:
- connect landmark pairs with green lines
- draw blue circles at visible joints
- annotate text overlays for form metrics such as `DEPTH`, `BODY`, `HIP`, `BALANCE`, and `EXT`

No-pose detection is also handled explicitly:

```python
if result.pose_landmarks:
    ...
else:
    self._draw_no_pose_warnings(image)
```

This gives the user immediate visual feedback if the body is not visible or the camera framing is poor.

## 8. Session flow and runtime behavior

### Login and initialization
On application startup:
- SQLite schema is created
- a login wall is rendered
- a default session state is initialized

### Workout planning
The sidebar lets the user choose:
- exercise type
- number of sets
- reps per set

Once the session starts, `workout_started` becomes true and app state is populated with:
- selected exercise
- target sets and rep count
- rep counter and set completion state
- timestamps and session metadata

### Video loop
Each incoming video frame is processed by `recv(self, frame)`, which:
- flips the image for mirror-correctness
- converts to MediaPipe image format
- calls `detect_for_video`
- draws skeleton and metrics
- updates metric state for the UI

### Feedback loop
The exercise detectors output structured metrics:

```python
{
    "reps": self.reps,
    "knee_angle": int(knee_angle),
    "back_angle": int(back_angle),
    "depth_status": depth_status,
}
```

These are interpreted by `VoicePipeline._find_form_issue(...)`, which converts them into coaching issues. If no issue is found, it can issue encouragement. If an issue is found and the cooldown has passed, the system asks the LLM for a response and converts it to speech.

## 9. AI coaching prompt design

The prompt is configured in `services/config/workout_config.py`.

The prompt tells the model to behave like a professional AI trainer and to return short coaching lines that are:
- natural and energetic
- around 10–15 words
- encouraging and technical where necessary
- aligned with exercise safety and good form

Example behavior:
- `workout_started` → motivational starting cue
- `set_completed` → positive acknowledgment
- `workout_completed` → closing encouragement
- `ongoing_form_check` + issue → corrective tip

This prompt design is essential because the LLM is not acting as a general chatbot; it is a constrained fitness coach operating on structured event data.

## 10. Audio pipeline details

The application uses a text-to-speech stage after the LLM generates a coaching sentence.

The audio pipeline is:
- `LLMCoach.give_feedback(...)` returns a sentence
- `TextToSpeech.speak(text)` converts it to spoken audio
- `autoplay_audio(audio_bytes)` calls the Streamlit audio widget with `autoplay=True`

This creates a near-real-time voice feedback loop that keeps the coach responsive during a live set.

## 11. Data model and persistence

The database stores user and workout data in `data.db`.

`users` table:
- `id`
- `username`
- `created_at`

`exercises` table:
- `id`
- `user_id`
- `exercise_name`
- `reps`
- `sets`
- `time`
- `created_at`

The repository logic ensures daily aggregation by exercise type and user id. This allows simple historical tracking without a heavier database stack.

## 12. Project structure

```text
AI-REALTIME-GYM-COACH/
├── .env
├── .streamlit/
├── core/
│   ├── __init__.py
│   └── base_exercise.py
├── data.db
├── detectors/
│   ├── biceps_curl.py
│   ├── lunges.py
│   ├── pushup.py
│   ├── shoulder_press.py
│   └── squat.py
├── guide/
├── main.py
├── ml_models/
│   └── pose_landmarker_full.task
├── packages.txt
├── requirements.txt
├── services/
│   ├── auth/
│   ├── coaching/
│   ├── config/
│   ├── persistence/
│   ├── state/
│   ├── static/
│   ├── tracking/
│   ├── ui/
│   └── vision/
└── README.md
```

## 13. Installation and setup

### Prerequisites
- Python 3.10+
- pip
- webcam access
- Groq API key

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables
Add your Groq API key to an environment variable or Streamlit secrets:

```bash
export GROQ_API_KEY="your_key_here"
```

Or via Streamlit secrets:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your_key_here"
```

The app also has a `.env` file in the project root, but in production you should keep this secret secure and avoid committing raw keys to version control.

### Run the app

```bash
streamlit run main.py
```

## 14. Runtime behavior notes

### Real-time requirements
This project is highly dependent on:
- camera quality and lighting
- proper user framing in the camera
- consistent body visibility for all landmarks
- stable background and minimal occlusion

### Accuracy assumptions
The detector logic is threshold-based rather than model-driven classification. That means:
- it works best when the user is facing the camera squarely
- proper side-view posture is essential for angle accuracy
- the system is suited for guided exercise repetition rather than precise clinical biomechanics analysis

### Why this approach matters
The architecture intentionally separates:
- pose estimation (MediaPipe)
- analytical rep counting (custom detectors)
- semantic coaching (Groq LLM)

This modular separation makes the system easier to extend with new exercises, more robust in live conditions, and suitable for interactive coaching rather than static offline inference.

## 15. Strengths of the implementation

- Real-time webcam feedback with annotated overlays
- Fast pose detection using MediaPipe
- Geometric angle-based detection without expensive custom model training
- Human-readable AI coaching feedback rather than raw metric dumps
- Hibernate-safe SQLite persistence for user history
- Audio feedback for hands-free training

## 16. Limitations and future improvements

Current constraints include:
- no advanced compensation for camera perspective distortion
- dependency on landmark visibility thresholds
- form feedback quality depends on the LLM prompt and the issue mapping layer
- limited exercise coverage compared to full fitness tracking platforms

Possible future improvements:
- add more exercises and custom rep-count logic
- integrate exercise-specific calibration or joint offset compensation
- support pose smoothing and temporal filtering
- add richer workout analytics dashboards
- support user authentication with stronger storage models
- add model versioning and A/B testing for coaching prompts
- support more robust voice command or low-latency audio generation

## 17. Summary

This project is a real-time AI fitness coach built around a Streamlit interface, MediaPipe pose landmark detection, custom geometric exercise logic, and a Groq-powered language model for coaching prompts. The critical ML computer-vision function is the MediaPipe pose landmark detector API: `vision.PoseLandmarker.create_from_options(...)` and `detect_for_video(...)`, using the model file `ml_models/pose_landmarker_full.task`.

The result is a practical computer-vision training assistant that does more than count reps: it reasons over body posture, detects form issues, and converts them into concise, supportive real-time coaching guidance.

## 18. Quick reference

### Key ML library usage
```python
from mediapipe.tasks.python import vision
self._landmarker = vision.PoseLandmarker.create_from_options(options)
result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
```

### Key LLM usage
```python
response = self.client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    temperature=0.4,
)
```

### Run command
```bash
streamlit run main.py
```

### Environment variable
```bash
GROQ_API_KEY="your_groq_key"
```

This README is intentionally technical and architecture-focused so developers can understand the system design, the computer-vision pipeline, and the AI coaching layer without needing to inspect every module individually.
