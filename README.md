---
title: PPE Detection System
emoji: 🦺
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.31.0"
app_file: app.py
pinned: false
---

# 🦺 PPE Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/YOUR_USERNAME/ppe-detection)

A real-time Personal Protective Equipment (PPE) detection system powered by YOLOv8, designed to identify safety compliance in workplace environments.

## 🎯 Features

- **📷 Image Detection**: Upload images for instant PPE analysis
- **🎥 Video Processing**: Analyze entire videos with frame-by-frame detection
- **📱 Live Stream Support**: Real-time detection via webcam or phone camera
- **🎨 Interactive UI**: Beautiful Streamlit interface with real-time visualization
- **📊 Detailed Analytics**: Comprehensive detection statistics and confidence scores

## 🏷️ Detected Classes

### ✅ PPE Present
- Glove
- Goggles
- Helmet
- Mask
- Suit
- Shoes

### ❌ PPE Missing
- No Glove
- No Goggles
- No Helmet
- No Mask
- No Suit
- No Shoes

## 📊 Model Performance

- **mAP@50**: 97.3%
- **mAP@50-95**: 69.2%
- **Precision**: 93.5%
- **Recall**: 93.3%
- **Model**: YOLOv8s (11.2M parameters)
- **Training**: 50 epochs on 4,979 validation images

## 🚀 Quick Start

### Option 1: Hugging Face Spaces (Recommended)

1. Visit the [live demo](https://huggingface.co/spaces/YOUR_USERNAME/ppe-detection)
2. Upload your trained `best.pt` model (or use the default)
3. Choose detection mode (Image/Video/Live Stream)
4. Start detecting!

### Option 2: Local Deployment

```bash
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/ppe-detection
cd ppe-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📱 Using Phone Camera

### Method 1: Direct Browser Access (Easiest)
1. Open the Hugging Face Space URL on your phone
2. Go to "Live Stream" tab
3. Use the camera input feature

### Method 2: IP Webcam App
1. Install **IP Webcam** (Android) or **EpocCam** (iOS)
2. Start the server in the app
3. Note the IP address (e.g., `http://192.168.1.100:8080`)
4. Enter the URL in the app's IP Camera option

### Method 3: Upload Frames
1. Take photos with your phone
2. Upload them in the "Upload Frame" section
3. Get instant detection results

## 📁 Project Structure

```
ppe-detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── best.pt               # Trained YOLOv8 model (upload separately)
└── .gitignore           # Git ignore rules
```

## 🔧 Configuration

### Model Upload
- Upload your trained `best.pt` model via the sidebar
- Or place it in the root directory as default

### Detection Settings
- **Confidence Threshold**: Adjust sensitivity (0.0 - 1.0)
- Default: 0.5 (50% confidence)

## 📊 Usage Examples

### Image Detection
```python
1. Upload an image (JPG, PNG)
2. View original vs annotated comparison
3. Check detection statistics
4. Download results
```

### Video Processing
```python
1. Upload a video (MP4, AVI, MOV)
2. Click "Process Video"
3. Wait for frame-by-frame analysis
4. Download annotated video
```

### Live Stream
```python
1. Choose camera source
2. Capture frames in real-time
3. Get instant PPE compliance feedback
```

## 🎨 Interface Preview

The app features:
- **Dual-column layout** for before/after comparison
- **Color-coded bounding boxes** (Green = PPE present, Red = PPE missing)
- **Confidence scores** for each detection
- **Real-time statistics** and analytics
- **Download options** for processed media

## 🔬 Technical Details

### Model Architecture
- **Base**: YOLOv8s (Small variant)
- **Input Size**: 640x640
- **Parameters**: 11.13M
- **GFLOPs**: 28.5

### Training Details
- **Dataset**: 12-class PPE detection
- **Training Time**: 2.84 hours (2x GPU T4)
- **Batch Size**: 32 (16 per GPU)
- **Optimizer**: AdamW

### Performance Metrics
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Suit | 100% | 93.9% | 99.3% | 98.7% |
| Goggles | 94.9% | 96.5% | 98.8% | 67.0% |
| Helmet | 95.1% | 94.6% | 97.5% | 61.5% |
| Overall | 93.5% | 93.3% | 97.3% | 69.2% |

## 🛠️ Deployment Options

### Option 1: GitHub Actions (Recommended - Auto Deploy)

**Setup once, deploy forever!** Every push to GitHub automatically deploys to Hugging Face.

#### Quick Setup (10 minutes):
1. **Create HF Token**: https://huggingface.co/settings/tokens (Write permission)
2. **Create HF Space**: https://huggingface.co/spaces → `ppe-detection` (Streamlit SDK)
3. **Create GitHub Repo**: https://github.com/new → `ppe-detection`
4. **Add GitHub Secrets**: 
   - Repo Settings → Secrets → Actions
   - Add `HF_TOKEN` (your HF token)
   - Add `HF_USERNAME` (your HF username)
5. **Push Code**: 
   ```bash
   git clone https://github.com/YOUR_USERNAME/ppe-detection.git
   cd ppe-detection
   # Copy files from safty folder
   git add .
   git commit -m "Initial commit"
   git push
   ```
6. **Auto-Deploy**: GitHub Actions deploys to HF automatically! 🚀

**Daily workflow:**
```bash
# Make changes
git add .
git commit -m "Update feature"
git push  # Auto-deploys to HF!
```

📚 **Detailed Guide**: See `GITHUB_ACTIONS_SETUP.md`  
⚡ **Quick Reference**: See `GITHUB_DEPLOY_QUICKREF.md`

---

### Option 2: Manual Deployment to Hugging Face

#### Step 1: Create a New Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose **Streamlit** as the SDK
4. Name it (e.g., `ppe-detection`)

#### Step 2: Upload Files
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ppe-detection
cd ppe-detection

# Copy your files
cp app.py .
cp requirements.txt .
cp packages.txt .
cp README.md .
cp -r .streamlit .
cp best.pt .  # Your trained model

# Push to Hugging Face
git add .
git commit -m "Initial commit"
git push
```

#### Step 3: Configure Space
1. Go to **Settings** in your Space
2. Set **Hardware**: CPU Basic (or GPU for faster inference)
3. Enable **Public** access
4. Save changes

#### Step 4: Access Your App
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/ppe-detection`
- Share the link with anyone!

## 📦 Model File

The trained model (`best.pt`) is **not included** in this repository due to size constraints.

### To use your model:
1. Download `best.pt` from your Kaggle output
2. Upload it via the Streamlit sidebar, OR
3. Place it in the root directory before deployment

### Model Size
- **File**: best.pt
- **Size**: ~22.5 MB
- **Format**: PyTorch (.pt)

## 🔒 Security & Privacy

- All processing happens **locally** or on Hugging Face servers
- No data is stored permanently
- Uploaded files are processed in temporary memory
- Videos and images are deleted after processing

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `best.pt` is in the correct format (YOLOv8 PyTorch model)
- Check file size (should be ~22-23 MB)
- Try re-uploading the model

### Video Processing Slow
- Large videos take time (1-2 minutes per minute of video)
- Consider upgrading to GPU hardware in Space settings
- Or process shorter clips

### Camera Not Working
- For local deployment: Check browser permissions
- For HF Spaces: Use "Upload Frame" method instead
- Ensure HTTPS connection for camera access

### NumPy Errors
- The app uses `numpy<2.0` to avoid compatibility issues
- If errors persist, try: `pip install numpy==1.26.4 --force-reinstall`

## 📈 Future Improvements

- [ ] Real-time video streaming (WebRTC)
- [ ] Multi-person tracking
- [ ] PPE compliance scoring
- [ ] Alert system for missing PPE
- [ ] Export detection logs (CSV/JSON)
- [ ] Custom model training interface
- [ ] Mobile app version

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **Streamlit** for the amazing framework
- **Hugging Face** for hosting
- **Roboflow** for dataset management

## 📞 Contact

For questions or support:
- Open an issue on GitHub
- Contact via Hugging Face Space discussions

---

**Built with ❤️ using YOLOv8 and Streamlit**

*Ensuring workplace safety through AI-powered PPE detection*
