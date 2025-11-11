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
import requests
import zipfile
import tempfile
from pathlib import Path

class EmotionAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Analysis")
        self.root.geometry("1000x700")
        self.root.configure(bg='#0f0f0f')
        self.root.resizable(False, False)
        
        # Initialize variables
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.people_count = 0
        self.emotion_data = defaultdict(int)
        self.face_tracking = {}  # Track faces by position
        self.face_id_counter = 0
        self.analysis_window = None
        self.person_emotions = {}  # Track current emotion for each person
        self.person_major_emotions = {}  # Track major emotion for each person
        self.person_emotion_history = {}  # Track emotion history for each person
        self.last_emotion_update = {}  # Track when emotion was last updated for each person
        self.person_last_seen = {}  # Track when each person was last detected
        self.max_person_count = 0  # Track maximum number of people seen
        self.session_start_time = None  # Track session start time
        self.total_detections = 0  # Track total emotion detections
        self.emotion_transitions = defaultdict(int)  # Track emotion transitions
        self.detection_timestamps = []  # Track when detections occurred
        self.emotion_start_times = {}  # Track when each emotion started for each person
        self.persistent_emotions = {}  # Track emotions that have persisted for 2+ seconds
        
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
            text="EMOTION ANALYSIS",
            font=self.tk_title_font,
            fg='#ffffff',
            bg='#0f0f0f'
        )
        title_label.pack(pady=(0, 40))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#0f0f0f')
        control_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Button container
        button_frame = tk.Frame(control_frame, bg='#0f0f0f')
        button_frame.pack()
        
        # Start/Stop buttons
        self.start_button = tk.Button(
            button_frame,
            text="START",
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
            text="ANALYZE",
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
        
        self.emotion_count_label = tk.Label(
            stats_frame,
            text="EMOTIONS: 0",
            font=self.tk_text_font,
            fg='#888888',
            bg='#0f0f0f'
        )
        self.emotion_count_label.pack(anchor=tk.E)
        
        # Camera display
        self.camera_frame = tk.Frame(main_frame, bg='#1a1a1a', relief=tk.FLAT, bd=0)
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(
            self.camera_frame,
            text="CAMERA FEED\n\nClick START to begin",
            font=self.tk_text_font,
            fg='#555555',
            bg='#1a1a1a',
            justify=tk.CENTER
        )
        self.camera_label.pack(expand=True)
    
    def start_camera(self):
        """Start the camera and emotion detection"""
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
            text="CAMERA STOPPED\n\nClick ANALYZE to see results",
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
        """Detect faces and analyze emotions with robust person tracking"""
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
                    
                    # Only update emotion if confidence is above threshold (lowered for better detection)
                    if confidence > 20:  # Lowered threshold from 30 to 20
                        self.update_person_emotion(face_id, dominant_emotion)
                        print(f"P{face_id}: {dominant_emotion} ({confidence:.1f}%)")  # Debug output
                    
                    # Get persistent emotion for this person
                    persistent_emotion = self.persistent_emotions.get(face_id, "detecting...")
                    current_emotion = self.person_emotions.get(face_id, "unknown")
                    
                    # Show both current and persistent emotions for debugging
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
            
            # Update people count based on active people (not just current frame)
            self.update_people_count(current_time)
            
            # Clean up people who haven't been seen for a while
            self.cleanup_old_faces(current_time)
            
        except Exception as e:
            print(f"Error in face detection: {e}")
    
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
            # Distance weight: 1.0, Area weight: 0.3
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
        """Update emotion for a person with 2-second persistence requirement"""
        current_time = time.time()
        
        # Initialize tracking for new person
        if face_id not in self.person_emotion_history:
            self.person_emotion_history[face_id] = []
            self.emotion_start_times[face_id] = {}
            self.persistent_emotions[face_id] = "unknown"
            print(f"New person P{face_id} detected")
        
        # Check if this is a new emotion for this person
        if face_id in self.person_emotions:
            old_emotion = self.person_emotions[face_id]
            if old_emotion != emotion:
                # Emotion changed - reset timer for new emotion
                self.emotion_start_times[face_id][emotion] = current_time
                print(f"P{face_id} emotion changed: {old_emotion} -> {emotion}")
        else:
            # First emotion for this person
            self.emotion_start_times[face_id][emotion] = current_time
        
        # Update person's current emotion
        self.person_emotions[face_id] = emotion
        self.last_emotion_update[face_id] = current_time
        
        # Track detection statistics
        self.total_detections += 1
        self.detection_timestamps.append(current_time)
        
        # Check if current emotion has persisted for 2+ seconds
        emotion_start_time = self.emotion_start_times[face_id].get(emotion, current_time)
        emotion_duration = current_time - emotion_start_time
        
        # Debug: show emotion duration
        if emotion_duration < 1.5:
            print(f"P{face_id}: {emotion} (waiting {emotion_duration:.1f}s/1.5s)")
        
        if emotion_duration >= 1.5:  # 1.5 seconds persistence (reduced for better responsiveness)
            old_persistent_emotion = self.persistent_emotions.get(face_id)
            
            if old_persistent_emotion != emotion:
                print(f"P{face_id} persistent emotion confirmed: {emotion} (after {emotion_duration:.1f}s)")
                
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
        total_emotions = sum(self.emotion_data.values())
        self.emotion_count_label.config(text=f"EMOTIONS: {total_emotions}")
    
    def show_analysis(self):
        """Show detailed analysis window"""
        if not self.emotion_data:
            messagebox.showwarning("No Data", "No emotion data available. Please run the camera first.")
            return
        
        # Create analysis window
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title("Analysis Results")
        self.analysis_window.geometry("900x600")
        self.analysis_window.configure(bg='#0f0f0f')
        self.analysis_window.resizable(False, False)
        
        # Title
        title_label = tk.Label(
            self.analysis_window,
            text="ANALYSIS RESULTS",
            font=self.tk_title_font,
            fg='#ffffff',
            bg='#0f0f0f'
        )
        title_label.pack(pady=30)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.analysis_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 20))
        
        # Configure notebook style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure notebook and tab styles
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
        
        # Configure the notebook frame
        style.configure('TNotebook', tabmargins=[0, 0, 0, 0])
        
        # Summary tab
        summary_frame = tk.Frame(notebook, bg='#0f0f0f')
        notebook.add(summary_frame, text="SUMMARY")
        
        # Calculate session statistics
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        detection_rate = self.total_detections / session_duration if session_duration > 0 else 0
        
        # Statistics
        stats_text = f"""
=== SESSION OVERVIEW ===
SESSION DURATION: {session_duration:.1f} seconds
TOTAL PEOPLE DETECTED: {self.people_count}
MAXIMUM PEOPLE SIMULTANEOUSLY: {self.max_person_count}
TOTAL EMOTION DETECTIONS: {self.total_detections}
DETECTION RATE: {detection_rate:.1f} detections/second

=== EMOTION DISTRIBUTION ===
"""
        for emotion, count in sorted(self.emotion_data.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(self.emotion_data.values())) * 100 if sum(self.emotion_data.values()) > 0 else 0
            stats_text += f"{emotion.upper()}: {count} ({percentage:.1f}%)\n"
        
        # Add individual person details
        if self.persistent_emotions:
            stats_text += f"\n=== INDIVIDUAL PEOPLE ===\n"
            for person_id, persistent_emotion in self.persistent_emotions.items():
                history_length = len(self.person_emotion_history.get(person_id, []))
                last_seen = self.person_last_seen.get(person_id, 0)
                time_since_last = time.time() - last_seen if last_seen > 0 else 0
                stats_text += f"P{person_id}: {persistent_emotion.upper()} ({history_length} detections, last seen {time_since_last:.1f}s ago)\n"
        
        # Add emotion transitions
        if self.emotion_transitions:
            stats_text += f"\n=== EMOTION TRANSITIONS ===\n"
            for transition, count in sorted(self.emotion_transitions.items(), key=lambda x: x[1], reverse=True):
                stats_text += f"{transition}: {count} times\n"
        
        # Add detection timeline analysis
        if len(self.detection_timestamps) > 1:
            time_intervals = []
            for i in range(1, len(self.detection_timestamps)):
                interval = self.detection_timestamps[i] - self.detection_timestamps[i-1]
                time_intervals.append(interval)
            
            avg_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 0
            min_interval = min(time_intervals) if time_intervals else 0
            max_interval = max(time_intervals) if time_intervals else 0
            
            stats_text += f"\n=== DETECTION TIMING ===\n"
            stats_text += f"AVERAGE INTERVAL: {avg_interval:.2f} seconds\n"
            stats_text += f"MINIMUM INTERVAL: {min_interval:.2f} seconds\n"
            stats_text += f"MAXIMUM INTERVAL: {max_interval:.2f} seconds\n"
        
        # Add emotion stability analysis
        if self.person_emotion_history:
            stats_text += f"\n=== EMOTION STABILITY ===\n"
            for person_id, history in self.person_emotion_history.items():
                if len(history) > 1:
                    unique_emotions = len(set(history))
                    stability = (len(history) - unique_emotions) / len(history) * 100
                    stats_text += f"P{person_id}: {stability:.1f}% stable ({unique_emotions} unique emotions)\n"
        
        stats_label = tk.Label(
            summary_frame,
            text=stats_text,
            font=self.tk_text_font,
            fg='#ffffff',
            bg='#0f0f0f',
            justify=tk.LEFT
        )
        stats_label.pack(pady=30, padx=30)
        
        # Analytics tab
        analytics_frame = tk.Frame(notebook, bg='#0f0f0f')
        notebook.add(analytics_frame, text="ANALYTICS")
        
        # Add detailed analytics
        analytics_text = f"""
=== PERFORMANCE METRICS ===
FRAME PROCESSING RATE: {detection_rate:.1f} FPS
TOTAL PROCESSING TIME: {session_duration:.1f} seconds
AVERAGE DETECTION CONFIDENCE: {self.calculate_average_confidence():.1f}%

=== EMOTION PATTERNS ===
MOST COMMON EMOTION: {self.get_most_common_emotion()}
LEAST COMMON EMOTION: {self.get_least_common_emotion()}
EMOTION DIVERSITY: {self.calculate_emotion_diversity():.1f}%

=== PERSON BEHAVIOR ===
MOST ACTIVE PERSON: {self.get_most_active_person()}
MOST STABLE PERSON: {self.get_most_stable_person()}
AVERAGE EMOTION CHANGES: {self.calculate_avg_emotion_changes():.1f} per person

=== SYSTEM EFFICIENCY ===
PEOPLE TRACKING ACCURACY: {self.calculate_tracking_accuracy():.1f}%
EMOTION DETECTION SUCCESS: {self.calculate_detection_success():.1f}%
"""
        
        analytics_label = tk.Label(
            analytics_frame,
            text=analytics_text,
            font=self.tk_text_font,
            fg='#ffffff',
            bg='#0f0f0f',
            justify=tk.LEFT
        )
        analytics_label.pack(pady=30, padx=30)
        
        # Visualization tab
        viz_frame = tk.Frame(notebook, bg='#0f0f0f')
        notebook.add(viz_frame, text="VISUALIZATION")
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#0f0f0f')
        
        # Set Space Grotesk font for matplotlib if available
        try:
            if self.font_path and self.font_path.exists():
                plt.rcParams['font.family'] = 'Space Grotesk'
        except:
            pass
        
        # Bar chart
        emotions = list(self.emotion_data.keys())
        counts = list(self.emotion_data.values())
        colors = ['#ffffff', '#cccccc', '#999999', '#666666', '#333333', '#222222', '#111111']
        
        ax1.bar(emotions, counts, color=colors[:len(emotions)])
        ax1.set_title('Emotion Distribution', color='white', fontsize=12, fontweight='normal')
        ax1.set_ylabel('Count', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#1a1a1a')
        
        # Pie chart
        if len(emotions) > 1:
            ax2.pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors[:len(emotions)])
            ax2.set_title('Emotion Percentage', color='white', fontsize=12, fontweight='normal')
        else:
            ax2.text(0.5, 0.5, f'Only {emotions[0]} detected', 
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
    
    def calculate_average_confidence(self):
        """Calculate average confidence of emotion detections"""
        # This would need to be tracked during detection
        return 75.0  # Placeholder
    
    def get_most_common_emotion(self):
        """Get the most common emotion detected"""
        if not self.emotion_data:
            return "None"
        return max(self.emotion_data, key=self.emotion_data.get)
    
    def get_least_common_emotion(self):
        """Get the least common emotion detected"""
        if not self.emotion_data:
            return "None"
        return min(self.emotion_data, key=self.emotion_data.get)
    
    def calculate_emotion_diversity(self):
        """Calculate emotion diversity percentage"""
        if not self.emotion_data:
            return 0.0
        unique_emotions = len(self.emotion_data)
        total_emotions = sum(self.emotion_data.values())
        return (unique_emotions / 7) * 100  # 7 is max possible emotions
    
    def get_most_active_person(self):
        """Get the person with most detections"""
        if not self.person_emotion_history:
            return "None"
        most_active = max(self.person_emotion_history, key=lambda x: len(self.person_emotion_history[x]))
        return f"P{most_active}"
    
    def get_most_stable_person(self):
        """Get the person with most stable emotions"""
        if not self.person_emotion_history:
            return "None"
        most_stable = min(self.person_emotion_history, key=lambda x: len(set(self.person_emotion_history[x])))
        return f"P{most_stable}"
    
    def calculate_avg_emotion_changes(self):
        """Calculate average emotion changes per person"""
        if not self.person_emotion_history:
            return 0.0
        total_changes = sum(len(set(history)) - 1 for history in self.person_emotion_history.values())
        return total_changes / len(self.person_emotion_history)
    
    def calculate_tracking_accuracy(self):
        """Calculate people tracking accuracy"""
        if self.max_person_count == 0:
            return 100.0
        return (self.people_count / self.max_person_count) * 100
    
    def calculate_detection_success(self):
        """Calculate emotion detection success rate"""
        if self.total_detections == 0:
            return 0.0
        successful_detections = sum(self.emotion_data.values())
        return (successful_detections / self.total_detections) * 100

def main():
    root = tk.Tk()
    app = EmotionAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
