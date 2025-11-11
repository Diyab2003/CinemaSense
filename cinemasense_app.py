import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image, ImageTk, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from collections import defaultdict, Counter
import os
from pathlib import Path

class CinemaSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CinemaSense - Movie Emotion Analysis")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0f0f0f')
        self.root.resizable(False, False)
        
        # Initialize variables
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.people_count = 0
        self.emotion_data = defaultdict(int)
        self.face_tracking = {}
        self.face_id_counter = 0
        self.analysis_window = None
        self.person_emotions = {}
        self.person_major_emotions = {}
        self.person_emotion_history = {}
        self.last_emotion_update = {}
        self.person_last_seen = {}
        self.max_person_count = 0
        self.session_start_time = None
        self.total_detections = 0
        self.emotion_transitions = defaultdict(int)
        self.detection_timestamps = []
        self.emotion_start_times = {}
        self.persistent_emotions = {}
        
        # CinemaSense specific variables
        self.movie_title = ""
        self.expected_emotion = ""
        self.reflection_scores = []
        self.match_count = 0
        self.total_checks = 0
        self.current_reflection_score = 0
        
        # Setup fonts
        self.setup_fonts()
        
        # Create GUI
        self.create_widgets()
        
        # Start the camera update loop
        self.update_camera()
        
    def setup_fonts(self):
        """Setup Space Grotesk font from local fonts directory"""
        self.font_path = self.get_space_grotesk_font()
        
        if self.font_path:
            try:
                # Load the Space Grotesk font
                self.title_font = ImageFont.truetype(str(self.font_path), 24)
                self.button_font = ImageFont.truetype(str(self.font_path), 12)
                self.text_font = ImageFont.truetype(str(self.font_path), 10)
                self.small_font = ImageFont.truetype(str(self.font_path), 8)
                
                # For tkinter widgets, we'll use the font name
                font_name = "Space Grotesk"
                self.tk_title_font = (font_name, 24, "bold")
                self.tk_button_font = (font_name, 12, "bold")
                self.tk_text_font = (font_name, 10)
                self.tk_small_font = (font_name, 8)
                
                print("Space Grotesk font loaded successfully!")
                
            except Exception as e:
                print(f"Error loading Space Grotesk font: {e}")
                self.setup_fallback_fonts()
        else:
            self.setup_fallback_fonts()
    
    def setup_fallback_fonts(self):
        """Setup fallback fonts if Space Grotesk is not available"""
        self.tk_title_font = ("Consolas", 24, "bold")
        self.tk_button_font = ("Consolas", 12, "bold")
        self.tk_text_font = ("Consolas", 10)
        self.tk_small_font = ("Consolas", 8)
        print("Using fallback fonts (Consolas)")
    
    def get_space_grotesk_font(self):
        """Get Space Grotesk font from local fonts directory"""
        fonts_dir = Path("fonts")
        
        # Check for Space Grotesk Variable font
        space_grotesk_font = fonts_dir / "SpaceGrotesk-VariableFont_wght.ttf"
        if space_grotesk_font.exists():
            print("Space Grotesk font found!")
            return space_grotesk_font
        
        print("Space Grotesk font not found in fonts directory")
        return None
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#0f0f0f')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="CINEMASENSE",
            font=self.tk_title_font,
            fg='#ffffff',
            bg='#0f0f0f'
        )
        title_label.pack(pady=(0, 20))
        
        # Movie configuration panel
        config_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.FLAT, bd=1)
        config_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Movie title input
        movie_frame = tk.Frame(config_frame, bg='#1a1a1a')
        movie_frame.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(movie_frame, text="MOVIE TITLE:", font=self.tk_text_font, 
                fg='#ffffff', bg='#1a1a1a').pack(side=tk.LEFT)
        
        self.movie_entry = tk.Entry(movie_frame, font=self.tk_text_font, 
                                   bg='#2a2a2a', fg='#ffffff', insertbackground='#ffffff',
                                   relief=tk.FLAT, bd=5)
        self.movie_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # Expected emotion selection
        emotion_frame = tk.Frame(config_frame, bg='#1a1a1a')
        emotion_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        tk.Label(emotion_frame, text="EXPECTED EMOTION:", font=self.tk_text_font, 
                fg='#ffffff', bg='#1a1a1a').pack(side=tk.LEFT)
        
        self.emotion_var = tk.StringVar(value="happy")
        emotion_combo = ttk.Combobox(emotion_frame, textvariable=self.emotion_var,
                                   values=["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"],
                                   font=self.tk_text_font, state="readonly")
        emotion_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#0f0f0f')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Button container
        button_frame = tk.Frame(control_frame, bg='#0f0f0f')
        button_frame.pack()
        
        # Start/Stop buttons
        self.start_button = tk.Button(
            button_frame,
            text="START ANALYSIS",
            font=self.tk_button_font,
            bg='#1a1a1a',
            fg='#ffffff',
            relief=tk.FLAT,
            padx=25,
            pady=12,
            command=self.start_camera,
            cursor='hand2',
            bd=0,
            highlightthickness=0
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        
        self.stop_button = tk.Button(
            button_frame,
            text="STOP",
            font=self.tk_button_font,
            bg='#2a2a2a',
            fg='#ffffff',
            relief=tk.FLAT,
            padx=25,
            pady=12,
            command=self.stop_camera,
            state=tk.DISABLED,
            cursor='hand2',
            bd=0,
            highlightthickness=0
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))
        
        # Analysis button
        self.analysis_button = tk.Button(
            button_frame,
            text="VIEW RESULTS",
            font=self.tk_button_font,
            bg='#1a1a1a',
            fg='#ffffff',
            relief=tk.FLAT,
            padx=25,
            pady=12,
            command=self.show_analysis,
            state=tk.DISABLED,
            cursor='hand2',
            bd=0,
            highlightthickness=0
        )
        self.analysis_button.pack(side=tk.LEFT)
        
        # Stats display
        stats_frame = tk.Frame(control_frame, bg='#0f0f0f')
        stats_frame.pack(side=tk.RIGHT)
        
        self.people_count_label = tk.Label(
            stats_frame,
            text="PEOPLE: 0",
            font=self.tk_text_font,
            fg='#888888',
            bg='#0f0f0f'
        )
        self.people_count_label.pack(anchor=tk.E)
        
        self.reflection_score_label = tk.Label(
            stats_frame,
            text="REFLECTION: 0%",
            font=self.tk_text_font,
            fg='#888888',
            bg='#0f0f0f'
        )
        self.reflection_score_label.pack(anchor=tk.E)
        
        # Camera display
        self.camera_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.FLAT, bd=0)
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(
            self.camera_frame,
            text="CAMERA FEED\n\nEnter movie details and click START ANALYSIS",
            font=self.tk_text_font,
            fg='#555555',
            bg='#1a1a1a',
            justify=tk.CENTER
        )
        self.camera_label.pack(expand=True)
    
    def start_camera(self):
        """Start the camera and emotion detection"""
        # Get movie configuration
        self.movie_title = self.movie_entry.get().strip()
        self.expected_emotion = self.emotion_var.get().strip()
        
        if not self.movie_title:
            messagebox.showwarning("Warning", "Please enter a movie title")
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.camera_active = True
            self.start_button.config(state=tk.DISABLED, bg='#2a2a2a')
            self.stop_button.config(state=tk.NORMAL, bg='#1a1a1a')
            self.analysis_button.config(state=tk.DISABLED, bg='#2a2a2a')
            
            # Reset counters
            self.people_count = 0
            self.emotion_data.clear()
            self.face_tracking.clear()
            self.face_id_counter = 0
            self.person_emotions.clear()
            self.person_major_emotions.clear()
            self.person_emotion_history.clear()
            self.last_emotion_update.clear()
            self.person_last_seen.clear()
            self.max_person_count = 0
            self.session_start_time = time.time()
            self.total_detections = 0
            self.emotion_transitions.clear()
            self.detection_timestamps.clear()
            self.emotion_start_times.clear()
            self.persistent_emotions.clear()
            
            # Reset CinemaSense specific variables
            self.reflection_scores.clear()
            self.match_count = 0
            self.total_checks = 0
            self.current_reflection_score = 0
            
            self.update_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera and enable analysis"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_button.config(state=tk.NORMAL, bg='#1a1a1a')
        self.stop_button.config(state=tk.DISABLED, bg='#2a2a2a')
        self.analysis_button.config(state=tk.NORMAL, bg='#1a1a1a')
        
        # Clear camera display
        self.camera_label.config(
            text="ANALYSIS STOPPED\n\nClick VIEW RESULTS to see reflection analysis",
            fg='#555555'
        )
    
    def update_camera(self):
        """Update camera feed and detect emotions"""
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces and emotions
                self.detect_faces_and_emotions(rgb_frame)
                
                # Convert to PIL Image and display
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.camera_label.config(image=photo, text="")
                self.camera_label.image = photo
                
                self.update_stats()
        
        # Schedule next update
        self.root.after(30, self.update_camera)
    
    def detect_faces_and_emotions(self, frame):
        """Detect faces and analyze emotions with reflection analysis"""
        try:
            # Convert to BGR for face detection
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            current_time = time.time()
            current_face_ids = []
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Find the closest existing face or assign new ID
                face_id = self.find_or_assign_face_id(x, y, w, h)
                current_face_ids.append(face_id)
                
                # Update last seen time for this person
                self.person_last_seen[face_id] = current_time
                
                # Extract face ROI
                face_roi = bgr_frame[y:y+h, x:x+w]
                
                try:
                    # Analyze emotion
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                    if isinstance(result, list):
                        emotion_data = result[0]['emotion']
                    else:
                        emotion_data = result['emotion']
                    
                    # Get dominant emotion
                    dominant_emotion = max(emotion_data, key=emotion_data.get)
                    confidence = emotion_data[dominant_emotion]
                    
                    # Only update emotion if confidence is above threshold
                    if confidence > 20:
                        self.update_person_emotion(face_id, dominant_emotion)
                        
                        # Calculate reflection score
                        self.calculate_reflection_score(dominant_emotion)
                    
                    # Get persistent emotion for this person
                    persistent_emotion = self.persistent_emotions.get(face_id, "detecting...")
                    current_emotion = self.person_emotions.get(face_id, "unknown")
                    
                    # Show both current and persistent emotions
                    if persistent_emotion == "detecting...":
                        display_text = f"P{face_id}: {current_emotion} (detecting...)"
                        color = (0, 255, 255)  # Yellow for detecting
                    else:
                        display_text = f"P{face_id}: {persistent_emotion}"
                        color = (0, 255, 0)  # Green for confirmed
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, display_text, 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                except Exception as e:
                    # Draw rectangle without emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"P{face_id}: Face detected", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Update people count based on active people
            self.update_people_count(current_time)
            
            # Clean up people who haven't been seen for a while
            self.cleanup_old_faces(current_time)
            
        except Exception as e:
            print(f"Error in face detection: {e}")
    
    def calculate_reflection_score(self, detected_emotion):
        """Calculate how well detected emotion reflects expected emotion"""
        self.total_checks += 1
        
        # Simple direct match
        if detected_emotion.lower() == self.expected_emotion.lower():
            self.match_count += 1
        
        # Calculate current reflection score
        if self.total_checks > 0:
            self.current_reflection_score = (self.match_count / self.total_checks) * 100
            self.reflection_scores.append(self.current_reflection_score)
    
    def update_people_count(self, current_time):
        """Update people count based on active people"""
        # Count people who have been seen recently (within last 2 seconds)
        active_people = 0
        for face_id, last_seen in self.person_last_seen.items():
            if current_time - last_seen < 2.0:  # Person seen within last 2 seconds
                active_people += 1
        
        # Update count and track maximum
        self.people_count = active_people
        if active_people > self.max_person_count:
            self.max_person_count = active_people
    
    def find_or_assign_face_id(self, x, y, w, h):
        """Find existing face ID or assign new one based on position and size"""
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        face_area = w * h
        
        best_match_id = None
        best_match_score = float('inf')
        
        # Check if this face matches any existing face
        for face_id, (prev_x, prev_y, prev_w, prev_h) in self.face_tracking.items():
            prev_center_x = prev_x + prev_w // 2
            prev_center_y = prev_y + prev_h // 2
            prev_area = prev_w * prev_h
            
            # Calculate distance between centers
            distance = ((face_center_x - prev_center_x) ** 2 + (face_center_y - prev_center_y) ** 2) ** 0.5
            
            # Calculate size similarity (area difference)
            area_diff = abs(face_area - prev_area) / max(face_area, prev_area)
            
            # Combined score (lower is better)
            score = distance + (area_diff * 100)
            
            # If this is a good match and better than previous matches
            if distance < 80 and area_diff < 0.5 and score < best_match_score:
                best_match_id = face_id
                best_match_score = score
        
        if best_match_id is not None:
            # Update position for existing person
            self.face_tracking[best_match_id] = (x, y, w, h)
            return best_match_id
        
        # New person - assign new ID
        new_id = self.face_id_counter
        self.face_id_counter += 1
        self.face_tracking[new_id] = (x, y, w, h)
        return new_id
    
    def update_person_emotion(self, face_id, emotion):
        """Update emotion for a person with persistence requirement"""
        current_time = time.time()
        
        # Initialize tracking for new person
        if face_id not in self.person_emotion_history:
            self.person_emotion_history[face_id] = []
            self.emotion_start_times[face_id] = {}
            self.persistent_emotions[face_id] = "unknown"
        
        # Check if this is a new emotion for this person
        if face_id in self.person_emotions:
            old_emotion = self.person_emotions[face_id]
            if old_emotion != emotion:
                # Emotion changed - reset timer for new emotion
                self.emotion_start_times[face_id][emotion] = current_time
        else:
            # First emotion for this person
            self.emotion_start_times[face_id][emotion] = current_time
        
        # Update person's current emotion
        self.person_emotions[face_id] = emotion
        self.last_emotion_update[face_id] = current_time
        
        # Track detection statistics
        self.total_detections += 1
        self.detection_timestamps.append(current_time)
        
        # Check if current emotion has persisted for 1.5+ seconds
        emotion_start_time = self.emotion_start_times[face_id].get(emotion, current_time)
        emotion_duration = current_time - emotion_start_time
        
        if emotion_duration >= 1.5:  # 1.5 seconds persistence
            old_persistent_emotion = self.persistent_emotions.get(face_id)
            
            if old_persistent_emotion != emotion:
                # Remove old persistent emotion from count
                if old_persistent_emotion and old_persistent_emotion != "unknown":
                    if self.emotion_data[old_persistent_emotion] > 0:
                        self.emotion_data[old_persistent_emotion] -= 1
                
                # Add new persistent emotion to count
                self.persistent_emotions[face_id] = emotion
                self.emotion_data[emotion] += 1
                
                # Track emotion transition
                if old_persistent_emotion and old_persistent_emotion != "unknown":
                    transition = f"{old_persistent_emotion} -> {emotion}"
                    self.emotion_transitions[transition] += 1
        
        # Add current emotion to history for analysis
        self.person_emotion_history[face_id].append(emotion)
        if len(self.person_emotion_history[face_id]) > 20:
            self.person_emotion_history[face_id].pop(0)
    
    def cleanup_old_faces(self, current_time):
        """Remove faces that haven't been seen for a while"""
        faces_to_remove = []
        
        # Remove people who haven't been seen for more than 5 seconds
        for face_id, last_seen in list(self.person_last_seen.items()):
            if current_time - last_seen > 5.0:  # 5 seconds timeout
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            # Remove from tracking
            if face_id in self.face_tracking:
                del self.face_tracking[face_id]
            
            # Remove persistent emotion count for this person
            if face_id in self.persistent_emotions:
                persistent_emotion = self.persistent_emotions[face_id]
                if persistent_emotion != "unknown" and self.emotion_data[persistent_emotion] > 0:
                    self.emotion_data[persistent_emotion] -= 1
                del self.persistent_emotions[face_id]
            
            # Remove from all tracking dictionaries
            if face_id in self.person_emotions:
                del self.person_emotions[face_id]
            if face_id in self.person_emotion_history:
                del self.person_emotion_history[face_id]
            if face_id in self.emotion_start_times:
                del self.emotion_start_times[face_id]
            if face_id in self.last_emotion_update:
                del self.last_emotion_update[face_id]
            if face_id in self.person_last_seen:
                del self.person_last_seen[face_id]
    
    def update_stats(self):
        """Update the statistics display"""
        self.people_count_label.config(text=f"PEOPLE: {self.people_count}")
        self.reflection_score_label.config(text=f"REFLECTION: {self.current_reflection_score:.1f}%")
    
    def show_analysis(self):
        """Show detailed reflection analysis"""
        if not self.emotion_data and self.total_checks == 0:
            messagebox.showwarning("No Data", "No emotion data available. Please run the analysis first.")
            return
        
        # Create analysis window
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title("CinemaSense - Reflection Analysis")
        self.analysis_window.geometry("1000x700")
        self.analysis_window.configure(bg='#0f0f0f')
        self.analysis_window.resizable(False, False)
        
        # Title
        title_label = tk.Label(
            self.analysis_window,
            text="REFLECTION ANALYSIS",
            font=self.tk_title_font,
            fg='#ffffff',
            bg='#0f0f0f'
        )
        title_label.pack(pady=30)
        
        # Movie info
        movie_info = tk.Label(
            self.analysis_window,
            text=f"MOVIE: {self.movie_title.upper()} | EXPECTED: {self.expected_emotion.upper()}",
            font=self.tk_text_font,
            fg='#888888',
            bg='#0f0f0f'
        )
        movie_info.pack(pady=(0, 20))
        
        # Main reflection score
        reflection_frame = tk.Frame(self.analysis_window, bg='#1a1a1a', relief=tk.FLAT, bd=1)
        reflection_frame.pack(fill=tk.X, padx=30, pady=(0, 20))
        
        score_color = self.get_reflection_color()
        score_label = tk.Label(
            reflection_frame,
            text=f"REFLECTION SCORE: {self.current_reflection_score:.1f}%",
            font=("Space Grotesk", 32, "bold"),
            fg=score_color,
            bg='#1a1a1a'
        )
        score_label.pack(pady=20)
        
        # Detailed metrics
        metrics_frame = tk.Frame(self.analysis_window, bg='#0f0f0f')
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(metrics_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Configure notebook style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#0f0f0f', borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background='#1a1a1a', 
                       foreground='#888888', 
                       font=self.tk_text_font,
                       padding=[25, 12],
                       borderwidth=0,
                       relief='flat')
        style.map('TNotebook.Tab',
                 background=[('selected', '#2a2a2a'), ('active', '#222222')],
                 foreground=[('selected', '#ffffff'), ('active', '#aaaaaa')])
        
        # Summary tab
        summary_frame = tk.Frame(notebook, bg='#0f0f0f')
        notebook.add(summary_frame, text="SUMMARY")
        
        # Calculate session statistics
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        # Statistics
        stats_text = f"""
=== SESSION OVERVIEW ===
MOVIE: {self.movie_title}
EXPECTED EMOTION: {self.expected_emotion}
SESSION DURATION: {session_duration:.1f} seconds
TOTAL PEOPLE DETECTED: {self.people_count}
MAXIMUM PEOPLE SIMULTANEOUSLY: {self.max_person_count}
TOTAL EMOTION CHECKS: {self.total_checks}

=== REFLECTION ANALYSIS ===
REFLECTION SCORE: {self.current_reflection_score:.1f}%
MATCHES: {self.match_count} / {self.total_checks}
SUCCESS RATE: {(self.match_count / self.total_checks * 100) if self.total_checks > 0 else 0:.1f}%

=== DETECTED EMOTIONS ===
"""
        for emotion, count in sorted(self.emotion_data.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(self.emotion_data.values())) * 100 if sum(self.emotion_data.values()) > 0 else 0
            match_indicator = "✓" if emotion.lower() == self.expected_emotion.lower() else "✗"
            stats_text += f"{emotion.upper()}: {count} ({percentage:.1f}%) {match_indicator}\n"
        
        # Add individual person details
        if self.persistent_emotions:
            stats_text += f"\n=== INDIVIDUAL PEOPLE ===\n"
            for person_id, persistent_emotion in self.persistent_emotions.items():
                history_length = len(self.person_emotion_history.get(person_id, []))
                last_seen = self.person_last_seen.get(person_id, 0)
                time_since_last = time.time() - last_seen if last_seen > 0 else 0
                match_indicator = "✓" if persistent_emotion.lower() == self.expected_emotion.lower() else "✗"
                stats_text += f"P{person_id}: {persistent_emotion.upper()} ({history_length} detections) {match_indicator}\n"
        
        # Add scrollable frame for summary
            summary_canvas = tk.Canvas(summary_frame, bg='#0f0f0f', highlightthickness=0)
            summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=summary_canvas.yview)
            scrollable_summary = tk.Frame(summary_canvas, bg='#0f0f0f')

            scrollable_summary.bind(
                "<Configure>",
                lambda e: summary_canvas.configure(
                    scrollregion=summary_canvas.bbox("all")
                )
            )

            summary_canvas.create_window((0, 0), window=scrollable_summary, anchor="nw")
            summary_canvas.configure(yscrollcommand=summary_scrollbar.set)

            summary_canvas.pack(side="left", fill="both", expand=True)
            summary_scrollbar.pack(side="right", fill="y")

            # Add stats label inside scrollable frame
            stats_label = tk.Label(
                scrollable_summary,
                text=stats_text,
                font=self.tk_text_font,
                fg='#ffffff',
                bg='#0f0f0f',
                justify=tk.LEFT,
                anchor="w"
            )
            stats_label.pack(pady=30, padx=30, anchor="w")

        
        # Visualization tab
        viz_frame = tk.Frame(notebook, bg='#0f0f0f')
        notebook.add(viz_frame, text="VISUALIZATION")
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0f0f0f')
        
        # Bar chart for detected emotions
        emotions = list(self.emotion_data.keys())
        counts = list(self.emotion_data.values())
        colors = []
        
        for emotion in emotions:
            if emotion.lower() == self.expected_emotion.lower():
                colors.append('#00ff00')  # Green for expected emotion
            else:
                colors.append('#ff4444')  # Red for other emotions
        
        if emotions:
            ax1.bar(emotions, counts, color=colors)
            ax1.set_title('Detected Emotions', color='white', fontsize=12, fontweight='normal')
            ax1.set_ylabel('Count', color='white')
            ax1.tick_params(colors='white')
            ax1.set_facecolor('#1a1a1a')
        
        # Reflection score over time
        if len(self.reflection_scores) > 1:
            ax2.plot(range(len(self.reflection_scores)), self.reflection_scores, 'b-', linewidth=2)
            ax2.set_title('Reflection Score Over Time', color='white', fontsize=12, fontweight='normal')
            ax2.set_ylabel('Reflection Score (%)', color='white')
            ax2.set_xlabel('Time', color='white')
            ax2.tick_params(colors='white')
            ax2.set_facecolor('#1a1a1a')
            ax2.set_ylim(0, 100)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for timeline', 
                    ha='center', va='center', color='white', fontsize=14)
            ax2.set_facecolor('#1a1a1a')
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Close button
        close_button = tk.Button(
            self.analysis_window,
            text="CLOSE",
            font=self.tk_button_font,
            bg='#1a1a1a',
            fg='#ffffff',
            relief=tk.FLAT,
            padx=25,
            pady=10,
            command=self.analysis_window.destroy,
            cursor='hand2',
            bd=0,
            highlightthickness=0
        )
        close_button.pack(pady=(0, 30))
    
    def get_reflection_color(self):
        """Get color based on reflection score"""
        if self.current_reflection_score >= 80:
            return '#00ff00'  # Green
        elif self.current_reflection_score >= 60:
            return '#ffff00'  # Yellow
        elif self.current_reflection_score >= 40:
            return '#ff8800'  # Orange
        else:
            return '#ff0000'  # Red

def main():
    root = tk.Tk()
    app = CinemaSenseApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
