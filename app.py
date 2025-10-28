import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import time
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF5252;
    }
    .detection-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# PPE Classes
PPE_CLASSES = {
    'glove': '✅ Glove',
    'goggles': '✅ Goggles',
    'helmet': '✅ Helmet',
    'mask': '✅ Mask',
    'suit': '✅ Suit',
    'shoes': '✅ Shoes',
    'no_glove': '❌ No Glove',
    'no_goggles': '❌ No Goggles',
    'no_helmet': '❌ No Helmet',
    'no_mask': '❌ No Mask',
    'no-suit': '❌ No Suit',
    'no_shoes': '❌ No Shoes'
}

# Color mapping for bounding boxes
COLORS = {
    'glove': (0, 255, 0),
    'goggles': (0, 255, 0),
    'helmet': (0, 255, 0),
    'mask': (0, 255, 0),
    'suit': (0, 255, 0),
    'shoes': (0, 255, 0),
    'no_glove': (0, 0, 255),
    'no_goggles': (0, 0, 255),
    'no_helmet': (0, 0, 255),
    'no_mask': (0, 0, 255),
    'no-suit': (0, 0, 255),
    'no_shoes': (0, 0, 255)
}

@st.cache_resource
def download_model_from_release(repo_owner, repo_name, tag="latest", model_filename="best.pt"):
    """Download model from GitHub Releases"""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / model_filename
    
    # If model already exists, return it
    if model_path.exists():
        return str(model_path)
    
    try:
        # Get latest release info
        if tag == "latest":
            api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
        else:
            api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/tags/{tag}"
        
        response = requests.get(api_url)
        response.raise_for_status()
        release_data = response.json()
        
        # Find the model asset
        model_asset = None
        for asset in release_data.get('assets', []):
            if asset['name'] == model_filename:
                model_asset = asset
                break
        
        if not model_asset:
            st.error(f"Model file '{model_filename}' not found in release assets")
            return None
        
        # Download the model
        download_url = model_asset['browser_download_url']
        st.info(f"📥 Downloading model from GitHub Release...")
        
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                progress_bar = st.progress(0)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_bar.progress(downloaded / total_size)
                progress_bar.empty()
        
        st.success(f"✅ Model downloaded successfully!")
        return str(model_path)
        
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def draw_detections(image, results, conf_threshold):
    """Draw bounding boxes and labels on image"""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            if conf >= conf_threshold:
                # Get class name
                class_name = result.names[cls]
                
                # Get color
                color = COLORS.get(class_name, (255, 255, 0))
                
                # Draw rectangle
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{PPE_CLASSES.get(class_name, class_name)}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), color, -1)
                cv2.putText(img, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img), detections

def process_video(video_path, model, conf_threshold, progress_bar, status_text):
    """Process video file and return annotated video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Draw detections
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame, detections = draw_detections(Image.fromarray(frame_rgb), results, conf_threshold)
        annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
        
        out.write(annotated_frame)
        all_detections.extend(detections)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path, all_detections

def main():
    # Header
    st.markdown('<p class="main-header">🦺 PPE Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect Personal Protective Equipment in Images, Videos & Live Stream</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model upload/selection
        st.subheader("1. Model Selection")
        model_option = st.radio(
            "Choose model source:",
            ["Download from GitHub Release", "Upload Model", "Use Local (if available)"]
        )
        
        if model_option == "Download from GitHub Release":
            with st.expander("⚙️ GitHub Release Settings", expanded=True):
                repo_owner = st.text_input(
                    "GitHub Username",
                    value="",
                    placeholder="your-username",
                    help="Your GitHub username"
                )
                repo_name = st.text_input(
                    "Repository Name",
                    value="safty",
                    help="Your repository name"
                )
                release_tag = st.text_input(
                    "Release Tag",
                    value="latest",
                    help="Release tag or 'latest' for the latest release"
                )
                model_filename = st.text_input(
                    "Model Filename",
                    value="best.pt",
                    help="Name of the model file in the release"
                )
                
                if st.button("📥 Download Model", type="primary"):
                    if repo_owner:
                        model_path = download_model_from_release(
                            repo_owner=repo_owner,
                            repo_name=repo_name,
                            tag=release_tag,
                            model_filename=model_filename
                        )
                        if model_path:
                            st.session_state.model = load_model(model_path)
                            st.success("✅ Model loaded successfully!")
                    else:
                        st.error("Please enter your GitHub username")
        
        elif model_option == "Upload Model":
            uploaded_model = st.file_uploader(
                "Upload your best.pt model",
                type=['pt'],
                help="Upload the trained YOLOv8 model file"
            )
            if uploaded_model:
                # Save uploaded model temporarily
                model_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pt').name
                with open(model_path, 'wb') as f:
                    f.write(uploaded_model.read())
                st.session_state.model = load_model(model_path)
                st.success("✅ Model loaded successfully!")
        
        else:  # Use Local
            # Check if local model exists
            local_paths = ['best.pt', 'models/best.pt', 'model/best.pt']
            model_found = False
            for path in local_paths:
                if os.path.exists(path):
                    st.session_state.model = load_model(path)
                    st.success(f"✅ Local model loaded from {path}!")
                    model_found = True
                    break
            
            if not model_found:
                st.warning("⚠️ No local model found. Please download or upload a model.")
        
        st.divider()
        
        # Detection settings
        st.subheader("2. Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        st.divider()
        
        # About
        st.subheader("📊 Model Info")
        st.info("""
        **Classes Detected:**
        - ✅ Glove, Goggles, Helmet
        - ✅ Mask, Suit, Shoes
        - ❌ Missing PPE items
        
        **Performance:**
        - mAP50: 97.3%
        - mAP50-95: 69.2%
        """)
    
    # Main content
    if st.session_state.model is None:
        st.warning("⚠️ Please upload or select a model from the sidebar to begin.")
        return
    
    # Tabs for different input modes
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📱 Live Stream"])
    
    # Tab 1: Image Detection
    with tab1:
        st.header("Upload Image for Detection")
        
        uploaded_image = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            key="image_uploader"
        )
        
        col1, col2 = st.columns(2)
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with st.spinner("🔍 Detecting PPE..."):
                # Run inference
                results = st.session_state.model(image, conf=conf_threshold, verbose=False)
                annotated_image, detections = draw_detections(image, results, conf_threshold)
            
            with col2:
                st.subheader("Detection Results")
                st.image(annotated_image, use_container_width=True)
            
            # Display detection statistics
            if detections:
                st.markdown("### 📊 Detection Summary")
                
                # Count detections by class
                detection_counts = {}
                for det in detections:
                    class_name = det['class']
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                
                # Display in columns
                cols = st.columns(4)
                for idx, (class_name, count) in enumerate(detection_counts.items()):
                    with cols[idx % 4]:
                        emoji = "✅" if not class_name.startswith('no') else "❌"
                        st.metric(
                            label=f"{emoji} {class_name.replace('_', ' ').title()}",
                            value=count
                        )
                
                # Detailed detections
                with st.expander("📋 Detailed Detections"):
                    for idx, det in enumerate(detections, 1):
                        st.write(f"**Detection {idx}:** {PPE_CLASSES.get(det['class'], det['class'])} "
                                f"(Confidence: {det['confidence']:.2%})")
            else:
                st.info("No PPE detected in the image.")
    
    # Tab 2: Video Detection
    with tab2:
        st.header("Upload Video for Detection")
        
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video_uploader"
        )
        
        if uploaded_video:
            # Save uploaded video temporarily
            video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            with open(video_path, 'wb') as f:
                f.write(uploaded_video.read())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                st.video(video_path)
            
            if st.button("🚀 Process Video", key="process_video_btn"):
                with col2:
                    st.subheader("Processing...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process video
                    output_path, detections = process_video(
                        video_path,
                        st.session_state.model,
                        conf_threshold,
                        progress_bar,
                        status_text
                    )
                    
                    status_text.text("✅ Processing complete!")
                    
                    st.subheader("Annotated Video")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=f,
                            file_name="ppe_detection_output.mp4",
                            mime="video/mp4"
                        )
                
                # Display statistics
                if detections:
                    st.markdown("### 📊 Video Detection Summary")
                    
                    detection_counts = {}
                    for det in detections:
                        class_name = det['class']
                        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    cols = st.columns(4)
                    for idx, (class_name, count) in enumerate(detection_counts.items()):
                        with cols[idx % 4]:
                            emoji = "✅" if not class_name.startswith('no') else "❌"
                            st.metric(
                                label=f"{emoji} {class_name.replace('_', ' ').title()}",
                                value=count
                            )
    
    # Tab 3: Live Stream
    with tab3:
        st.header("Live Camera Stream Detection")
        
        st.info("""
        ### 📱 How to use Live Stream:
        
        **Option 1: Desktop Webcam**
        - Click "Start Webcam" below
        - Allow camera access when prompted
        
        **Option 2: Phone Camera**
        1. Install **DroidCam** or **IP Webcam** app on your phone
        2. Connect phone and computer to same WiFi
        3. Use the phone's IP address as camera source
        4. Or use Hugging Face Spaces on your phone browser
        
        **Option 3: Phone Browser (Recommended for HF Spaces)**
        - Open this app on your phone's browser
        - Use the webcam feature (will access phone camera)
        """)
        
        # Camera options
        camera_option = st.radio(
            "Select Camera Source:",
            ["Webcam", "IP Camera (Phone)", "Upload Frame"]
        )
        
        if camera_option == "Webcam":
            st.warning("⚠️ Webcam streaming requires running locally or using Streamlit's camera_input component.")
            
            # Use Streamlit's camera input
            camera_image = st.camera_input("Take a picture")
            
            if camera_image:
                image = Image.open(camera_image)
                
                with st.spinner("🔍 Detecting PPE..."):
                    results = st.session_state.model(image, conf=conf_threshold, verbose=False)
                    annotated_image, detections = draw_detections(image, results, conf_threshold)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Captured Frame")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    st.image(annotated_image, use_container_width=True)
                
                if detections:
                    st.markdown("### 📊 Detection Summary")
                    detection_counts = {}
                    for det in detections:
                        class_name = det['class']
                        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    cols = st.columns(4)
                    for idx, (class_name, count) in enumerate(detection_counts.items()):
                        with cols[idx % 4]:
                            emoji = "✅" if not class_name.startswith('no') else "❌"
                            st.metric(
                                label=f"{emoji} {class_name.replace('_', ' ').title()}",
                                value=count
                            )
        
        elif camera_option == "IP Camera (Phone)":
            st.markdown("""
            ### 📱 Setup IP Camera from Phone:
            
            1. **Install IP Webcam App** (Android) or **EpocCam** (iOS)
            2. **Start Server** in the app
            3. **Note the IP address** shown (e.g., http://192.168.1.100:8080)
            4. **Enter the URL below**
            """)
            
            ip_url = st.text_input(
                "Enter IP Camera URL:",
                placeholder="http://192.168.1.100:8080/video",
                help="Full URL to the video stream"
            )
            
            if ip_url and st.button("🎥 Start Stream"):
                st.warning("⚠️ IP camera streaming requires local deployment. For Hugging Face Spaces, use 'Upload Frame' option.")
        
        else:  # Upload Frame
            st.markdown("### 📸 Upload Frame from Phone")
            st.info("Take a photo with your phone and upload it here for instant detection!")
            
            frame_upload = st.file_uploader(
                "Upload a frame/photo",
                type=['jpg', 'jpeg', 'png'],
                key="frame_uploader"
            )
            
            if frame_upload:
                image = Image.open(frame_upload)
                
                with st.spinner("🔍 Detecting PPE..."):
                    results = st.session_state.model(image, conf=conf_threshold, verbose=False)
                    annotated_image, detections = draw_detections(image, results, conf_threshold)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Uploaded Frame")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    st.image(annotated_image, use_container_width=True)
                
                if detections:
                    st.markdown("### 📊 Detection Summary")
                    detection_counts = {}
                    for det in detections:
                        class_name = det['class']
                        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    cols = st.columns(4)
                    for idx, (class_name, count) in enumerate(detection_counts.items()):
                        with cols[idx % 4]:
                            emoji = "✅" if not class_name.startswith('no') else "❌"
                            st.metric(
                                label=f"{emoji} {class_name.replace('_', ' ').title()}",
                                value=count
                            )

if __name__ == "__main__":
    main()
