# Voice Recommendation System - Frontend

A modern, responsive web interface for the Voice Recommendation System that allows users to upload audio files or record audio using their microphone to get intelligent recommendations.

## Features

- **Audio Upload**: Drag-and-drop or click to upload audio files
- **Microphone Recording**: Record audio directly in the browser
- **Real-time Processing**: Live feedback during audio processing
- **Multilingual Support**: Displays detected language and transcription
- **Translation**: Shows translation from detected language to English
- **Smart Recommendations**: Displays relevant Q&A matches
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Beautiful, intuitive interface with smooth animations

## Supported Audio Formats

- MP3, WAV, FLAC, OGG, M4A, AAC, MP4
- WMA, AMR, AIFF, AU, 3GP, WebM, MPEG
- Maximum file size: 50MB

## Quick Start
cd /workspace/voice_recommendation_system/frontend && python3 -m http.server 8013
### Prerequisites

1. **Backend API Running**: Ensure the Voice Recommendation System backend is running on `http://localhost:5000`
2. **Modern Browser**: Chrome, Firefox, Safari, or Edge with Web Audio API support
3. **Microphone Access**: For recording functionality (browser will request permission)

### Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd /workspace/voice_recommendation_system/frontend
   ```

2. **Serve the files using a local web server:**

   **Option 1: Python HTTP Server**
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Python 2
   python -m SimpleHTTPServer 8000
   ```

   **Option 2: Node.js HTTP Server**
   ```bash
   npx http-server -p 8000
   ```

   **Option 3: PHP Server**
   ```bash
   php -S localhost:8000
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:8000
   ```

## Usage

### Upload Audio File

1. Click on the "Upload Audio" tab
2. Either:
   - Drag and drop an audio file onto the upload area
   - Click "Browse Files" to select a file
3. Click "Process Audio" to analyze the file

### Record Audio

1. Click on the "Record Audio" tab
2. Click "Start Recording" (browser will request microphone permission)
3. Speak into your microphone
4. Click "Stop Recording" when finished
5. Optionally click "Play" to review your recording
6. Click "Process Audio" to analyze the recording

### View Results

After processing, you'll see:

- **Language Detection**: Detected language with confidence level
- **Transcription**: Original text with character and word counts
- **Translation**: English translation (if applicable)
- **Recommended Answers**: Relevant Q&A matches with similarity scores

## API Integration

The frontend communicates with the backend API through the following endpoints:

- `GET /health` - Check API status
- `POST /process-audio` - Complete audio processing workflow
- `POST /transcribe` - Audio transcription only
- `POST /search` - Text-based semantic search

## Browser Compatibility

### Required Features

- **Web Audio API**: For microphone recording
- **File API**: For file uploads
- **Fetch API**: For HTTP requests
- **ES6 Classes**: For JavaScript functionality

### Supported Browsers

- Chrome 60+
- Firefox 55+
- Safari 11+
- Edge 79+

### Mobile Support

- iOS Safari 11+
- Chrome Mobile 60+
- Firefox Mobile 55+

## File Structure

```
frontend/
├── index.html          # Main HTML file
├── styles.css          # CSS styles and responsive design
├── script.js           # JavaScript functionality
└── README.md           # This file
```

## Customization

### Styling

Edit `styles.css` to customize:
- Colors and themes
- Layout and spacing
- Animations and transitions
- Responsive breakpoints

### API Configuration

In `script.js`, modify the `apiBaseUrl` to point to your backend:

```javascript
this.apiBaseUrl = 'http://your-backend-url:5000';
```

### Features

You can disable certain features by modifying the HTML:
- Remove recording tab to disable microphone functionality
- Modify file input `accept` attribute to restrict file types
- Adjust file size limits in the validation logic

## Troubleshooting

### Common Issues

1. **"Cannot connect to API"**
   - Ensure the backend is running on `http://localhost:5000`
   - Check for CORS issues if using a different domain
   - Verify firewall settings

2. **"Unable to access microphone"**
   - Grant microphone permissions in your browser
   - Ensure you're using HTTPS (required for microphone access on some browsers)
   - Check if another application is using the microphone

3. **"File type not supported"**
   - Ensure the file is a valid audio format
   - Check file size (must be under 50MB)
   - Try converting the file to a more common format (MP3, WAV)

4. **Recording not working**
   - Use a modern browser with Web Audio API support
   - Ensure microphone is not being used by another application
   - Check browser console for error messages

### Browser Console

Open browser developer tools (F12) and check the Console tab for error messages that can help diagnose issues.

## Development

### Adding New Features

1. **New API Endpoints**: Add methods to the `VoiceRecommendationApp` class
2. **UI Components**: Add HTML structure and corresponding CSS styles
3. **Event Handlers**: Add event listeners in the `initializeEventListeners` method

### Testing

1. Test with different audio formats and sizes
2. Verify responsive design on various screen sizes
3. Test microphone recording on different devices
4. Check error handling with invalid inputs

## Security Considerations

- The frontend runs entirely in the browser
- No sensitive data is stored locally
- Audio files are processed and immediately discarded
- API communication uses standard HTTP (consider HTTPS for production)

## Performance

- Audio files are processed in memory (not stored on disk)
- Large files may take longer to process
- Recording quality affects processing time
- Results are cached in the browser session

## License

This frontend is part of the Voice Recommendation System. See the main repository for license information.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review browser console for error messages
3. Ensure backend API is running and accessible
4. Verify browser compatibility requirements
