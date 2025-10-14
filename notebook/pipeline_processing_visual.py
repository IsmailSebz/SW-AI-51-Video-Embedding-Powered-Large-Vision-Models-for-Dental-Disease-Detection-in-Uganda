#!/usr/bin/env python3
"""
Unified Video Reasoning Pipeline with DeepSeek via Ollama - FIXED VERSION
Author: Research Assistant
Description: Processes video through YOLO for real-time detection and uses DeepSeek via Ollama for reasoning
"""

import cv2
import time
import json
import subprocess
import threading
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional
import requests
import numpy as np
from ultralytics import YOLO
import os
import queue


class UnifiedVideoReasoner:
    def __init__(self, yolo_model: str = "./models/runs/detect/train/weights/best.pt",
                 ollama_base_url: str = "http://localhost:11434",
                 deepseek_model: str = "deepseek-r1:1.5b",
                 show_preview: bool = True):
        """
        Initialize the unified video reasoning pipeline

        Args:
            yolo_model: Path to YOLO model or model name
            ollama_base_url: Base URL for Ollama API
            deepseek_model: DeepSeek model name in Ollama
            show_preview: Whether to show real-time preview (can cause issues on some systems)
        """
        # Initialize YOLO model
        print("🚀 Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model)

        # Ollama configuration
        self.ollama_url = ollama_base_url
        self.deepseek_model = deepseek_model
        self.show_preview = show_preview

        # State management
        self.is_recording = False
        self.is_processing = False

        # Data storage
        self.detection_history = defaultdict(list)
        self.frame_buffer = deque(maxlen=30)
        self.video_info = {}

        # Thread-safe communication
        self.preview_queue = queue.Queue(maxsize=1)
        self.stop_preview = threading.Event()

        # Verify Ollama connection and model
        self._verify_ollama_setup()

        print("✅ Unified Video Reasoner initialized successfully!")

    def _verify_ollama_setup(self):
        """Verify Ollama is running and DeepSeek model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise Exception("Ollama is not running. Please start Ollama first.")

            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]

            if self.deepseek_model not in model_names:
                print(f"⚠️  DeepSeek model '{self.deepseek_model}' not found in Ollama.")
                print("Available models:", [name for name in model_names if name])
                print(f"Please pull the model first: ollama pull {self.deepseek_model}")
                raise Exception(f"Model {self.deepseek_model} not available")

            print(f"✅ DeepSeek model '{self.deepseek_model}' is available")

        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        except requests.exceptions.Timeout:
            raise Exception("Ollama connection timeout. Make sure Ollama is running and accessible.")

    def _preview_thread(self):
        """Dedicated thread for handling OpenCV preview window"""
        print("👀 Starting preview thread...")
        window_name = 'Recording - Press q to stop early'

        while not self.stop_preview.is_set():
            try:
                # Get frame from queue with timeout
                frame_data = self.preview_queue.get(timeout=0.1)
                if frame_data is None:
                    break

                frame, frame_count = frame_data

                # Display frame
                cv2.imshow(window_name, frame)

                # Check for key press (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("⏹️  Stopping recording via user input...")
                    self.is_recording = False
                    break

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Preview error: {e}")
                break

        # Cleanup
        cv2.destroyAllWindows()
        print("✅ Preview thread stopped")

    def _call_deepseek(self, prompt: str, max_tokens: int = 512) -> str:
        """Call DeepSeek model through Ollama API"""
        payload = {
            "model": self.deepseek_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json().get('response', '').strip()

        except requests.exceptions.RequestException as e:
            return f"Error calling DeepSeek: {str(e)}"

    def start_recording(self, video_source: Any = 0, duration: int = 10,
                        output_path: str = "recorded_video.mp4") -> None:
        """Start recording video with real-time YOLO processing"""
        if self.is_recording:
            print("⚠️  Recording already in progress")
            return

        print(f"🎥 Starting recording for {duration} seconds...")
        self.is_recording = True
        self.stop_preview.clear()
        self.detection_history.clear()
        self.frame_buffer.clear()

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise Exception(f"Could not open video source: {video_source}")

        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        self.video_info = {
            'width': frame_width,
            'height': frame_height,
            'fps': fps,
            'duration': duration
        }

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Start preview thread if enabled
        if self.show_preview:
            self.preview_thread = threading.Thread(target=self._preview_thread)
            self.preview_thread.daemon = True
            self.preview_thread.start()

        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._record_video,
            args=(duration,)
        )
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def _record_video(self, duration: int) -> None:
        """Record video with real-time YOLO processing (MAIN FIX)"""
        start_time = time.time()
        frame_count = 0

        try:
            while self.is_recording and (time.time() - start_time) < duration:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from video source")
                    break

                # Store frame
                self.frame_buffer.append({
                    'frame': frame.copy(),
                    'timestamp': time.time() - start_time,
                    'frame_id': frame_count
                })

                # Run YOLO inference
                try:
                    results = self.yolo_model(frame, verbose=False, conf=0.5)
                except Exception as e:
                    print(f"❌ YOLO inference error: {e}")
                    break

                # Process detections
                detections = []
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = results[0].names[cls]

                        detection = {
                            'frame_id': frame_count,
                            'timestamp': time.time() - start_time,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class_id': cls
                        }
                        detections.append(detection)

                        # Store in history
                        self.detection_history[class_name].append({
                            'frame_id': frame_count,
                            'timestamp': time.time() - start_time,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf)
                        })

                # Create annotated frame
                try:
                    annotated_frame = results[0].plot()

                    # Write to video file
                    self.out.write(annotated_frame)

                    # Send to preview thread if enabled
                    if self.show_preview and not self.preview_queue.full():
                        try:
                            self.preview_queue.put((annotated_frame, frame_count), timeout=0.01)
                        except queue.Full:
                            pass  # Skip frame if queue is full

                except Exception as e:
                    print(f"⚠️  Frame annotation error: {e}")
                    # Still write original frame if annotation fails
                    self.out.write(frame)

                frame_count += 1

                # Small delay to prevent overwhelming the system
                time.sleep(0.01)

        except Exception as e:
            print(f"❌ Recording error: {e}")

        finally:
            # Cleanup
            self.is_recording = False
            self.stop_preview.set()

            if hasattr(self, 'cap'):
                self.cap.release()
            if hasattr(self, 'out'):
                self.out.release()

            # Signal preview thread to stop
            if self.show_preview:
                try:
                    self.preview_queue.put(None, timeout=0.1)
                except queue.Full:
                    pass

            print(f"✅ Recording completed. Processed {frame_count} frames.")
            print(f"📊 Detected {len(self.detection_history)} object types")

    def stop_recording(self) -> None:
        """Stop recording manually"""
        if self.is_recording:
            self.is_recording = False
            self.stop_preview.set()
            print("⏹️  Recording stopped manually")

    def _create_detection_summary(self) -> Dict[str, Any]:
        """Create comprehensive summary of detected objects"""
        if not self.detection_history:
            return {"error": "No detection data available"}

        summary = []
        total_detections = 0

        for class_name, detections in self.detection_history.items():
            if detections:
                timestamps = [d['timestamp'] for d in detections]
                confidences = [d['confidence'] for d in detections]

                summary.append({
                    'object': class_name,
                    'count': len(detections),
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'duration': max(timestamps) - min(timestamps),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences)
                })
                total_detections += len(detections)

        # Sort by count (most frequent first)
        summary.sort(key=lambda x: x['count'], reverse=True)

        return {
            'total_objects_detected': total_detections,
            'unique_object_types': len(summary),
            'objects': summary,
            'recording_duration': self.video_info.get('duration', 0)
        }

    def _detect_activity_patterns(self) -> Dict[str, List[str]]:
        """Detect temporal patterns in object appearances"""
        patterns = {
            'continuous_objects': [],
            'transient_objects': [],
            'frequent_objects': [],
            'high_confidence_objects': []
        }

        for class_name, detections in self.detection_history.items():
            if len(detections) == 0:
                continue

            timestamps = [d['timestamp'] for d in detections]
            confidences = [d['confidence'] for d in detections]
            duration = max(timestamps) - min(timestamps)
            total_duration = self.video_info.get('duration', 10)

            if duration > total_duration * 0.7:
                patterns['continuous_objects'].append(class_name)

            if len(detections) < 3:
                patterns['transient_objects'].append(class_name)

            if len(detections) > 10:
                patterns['frequent_objects'].append(class_name)

            if np.mean(confidences) > 0.8:
                patterns['high_confidence_objects'].append(class_name)

        return patterns

    def analyze_video(self, question: str = None) -> Dict[str, Any]:
        """Analyze the recorded video using DeepSeek"""
        if self.is_recording:
            return {"error": "Recording in progress. Please stop recording first."}

        if not self.detection_history:
            return {"error": "No detection data available. Please record a video first."}

        print("🧠 Starting DeepSeek analysis...")
        self.is_processing = True

        try:
            detection_summary = self._create_detection_summary()
            activity_patterns = self._detect_activity_patterns()

            if question is None:
                question = "Based on the objects detected throughout the video, describe what was happening in the scene, the main activities, and any interesting patterns you observe."

            prompt = self._build_analysis_prompt(detection_summary, activity_patterns, question)

            print("🤔 DeepSeek is reasoning about the video...")
            start_time = time.time()
            analysis = self._call_deepseek(prompt)
            processing_time = time.time() - start_time

            results = {
                'analysis': analysis,
                'detection_summary': detection_summary,
                'activity_patterns': activity_patterns,
                'processing_time_seconds': round(processing_time, 2),
                'question': question
            }

            print(f"✅ Analysis completed in {processing_time:.2f} seconds")
            return results

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
        finally:
            self.is_processing = False

    def _build_analysis_prompt(self, detection_summary: Dict, activity_patterns: Dict, question: str) -> str:
        """Build comprehensive prompt for DeepSeek"""

        object_descriptions = []
        for obj in detection_summary.get('objects', []):
            desc = (f"- {obj['object']}: appeared {obj['count']} times, "
                    f"visible for {obj['duration']:.1f}s, "
                    f"confidence: {obj['avg_confidence']:.2f}")
            object_descriptions.append(desc)

        pattern_descriptions = []
        for pattern_type, objects in activity_patterns.items():
            if objects:
                readable_type = pattern_type.replace('_', ' ').title()
                pattern_descriptions.append(f"- {readable_type}: {', '.join(objects)}")

        prompt = f"""You are a video analysis expert. Analyze the object detection results from a video recording and provide insightful reasoning.

VIDEO CONTEXT:
Recording Duration: {detection_summary.get('recording_duration', 0)} seconds
Total Detections: {detection_summary.get('total_objects_detected', 0)}
Unique Object Types: {detection_summary.get('unique_object_types', 0)}

DETECTED OBJECTS:
{chr(10).join(object_descriptions)}

ACTIVITY PATTERNS:
{chr(10).join(pattern_descriptions) if pattern_descriptions else "- No significant patterns detected"}

ANALYSIS TASK: {question}

Please provide:
1. A concise summary of the main scene
2. Analysis of object interactions and activities
3. Notable patterns or anomalies
4. Answer to the specific question

Reason step by step and be factual based on the detection data:"""

        return prompt

    def interactive_analysis(self) -> None:
        """Run interactive analysis session"""
        if not self.detection_history:
            print("❌ No video data available. Please record a video first.")
            return

        print("\n" + "=" * 60)
        print("🤖 INTERACTIVE VIDEO ANALYSIS")
        print("=" * 60)

        print("\n📊 Generating initial analysis...")
        results = self.analyze_video()

        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return

        self._print_results(results)

        while True:
            print("\n" + "-" * 40)
            question = input("\n💭 Ask a question about the video (or 'quit' to exit): ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            print("🤔 Thinking...")
            results = self.analyze_video(question)

            if 'error' in results:
                print(f"❌ Error: {results['error']}")
            else:
                print(f"\n💡 DeepSeek's Answer:")
                print(f"{results['analysis']}")

    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted results"""
        print(f"\n📋 DETECTION SUMMARY:")
        print(f"   Total objects detected: {results['detection_summary']['total_objects_detected']}")
        print(f"   Unique object types: {results['detection_summary']['unique_object_types']}")

        print(f"\n🏃 ACTIVITY PATTERNS:")
        for pattern_type, objects in results['activity_patterns'].items():
            if objects:
                readable_type = pattern_type.replace('_', ' ').title()
                print(f"   {readable_type}: {', '.join(objects)}")

        print(f"\n🧠 DEEPSEEK ANALYSIS:")
        print(f"{results['analysis']}")

        print(f"\n⏱️  Processing time: {results['processing_time_seconds']}s")

    def save_analysis_report(self, results: Dict[str, Any], filename: str = "video_analysis_report.json") -> None:
        """Save analysis results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Analysis report saved to {filename}")


def main():
    """Main function demonstrating the pipeline"""
    # Initialize with preview disabled to avoid threading issues
    reasoner = UnifiedVideoReasoner(
        yolo_model="yolov8n.pt",
        ollama_base_url="http://localhost:11434",
        deepseek_model="deepseek-r1:1.5b",
        show_preview=True  # Disable preview to avoid threading issues
    )

    print("🎯 Unified Video Reasoning Pipeline")
    print("Options:")
    print("1. Record from webcam (no preview - safer)")
    print("2. Record from webcam (with preview - may cause issues)")
    print("3. Record from video file")
    print("4. Analyze existing recording")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        # Webcam recording without preview
        reasoner.show_preview = False
        duration = int(input("Enter recording duration in seconds (default 10): ") or "10")
        reasoner.start_recording(video_source=0, duration=duration)
        print("📹 Recording... (press Ctrl+C to stop early)")

    elif choice == "2":
        # Webcam recording with preview
        reasoner.show_preview = True
        duration = int(input("Enter recording duration in seconds (default 10): ") or "10")
        reasoner.start_recording(video_source=0, duration=duration)
        print("📹 Recording... (press 'q' in preview window to stop early)")

    elif choice == "3":
        # Video file processing
        video_file = input("Enter path to video file: ").strip()
        if not os.path.exists(video_file):
            print("❌ Video file not found!")
            return

        print("📹 Processing video file...")
        reasoner.show_preview = False
        reasoner.start_recording(video_source=video_file, duration=3600)
        time.sleep(2)
        reasoner.stop_recording()

    elif choice == "4":
        # Analyze existing data
        pass
    else:
        print("❌ Invalid choice!")
        return

    # Wait for recording to complete
    if choice in ['1', '2', '3']:
        while reasoner.is_recording:
            time.sleep(1)
        print("✅ Recording finished!")

    # Perform analysis
    if reasoner.detection_history:
        print("\n" + "=" * 50)
        custom_question = input("Enter your analysis question (or press Enter for general analysis): ").strip()

        results = reasoner.analyze_video(custom_question if custom_question else None)

        if 'error' not in results:
            reasoner._print_results(results)

            save = input("\n💾 Save analysis report? (y/n): ").strip().lower()
            if save == 'y':
                filename = input("Enter filename (default: video_analysis_report.json): ").strip()
                reasoner.save_analysis_report(results, filename or "video_analysis_report.json")

            interactive = input("\n🔍 Start interactive Q&A session? (y/n): ").strip().lower()
            if interactive == 'y':
                reasoner.interactive_analysis()
    else:
        print("❌ No detection data available for analysis.")


if __name__ == "__main__":
    main()