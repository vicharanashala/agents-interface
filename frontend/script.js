// Voice Recommendation System Frontend
class VoiceRecommendationApp {
    constructor() {
        console.log('=== SIMPLIFIED UI VERSION LOADED ===');
        this.apiBaseUrl = CONFIG.API.BASE_URL;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.recordingStartTime = null;
        this.recordingTimer = null;
        this.isRecording = false;
        this.audioBlob = null;
        
        // Live processing properties
        this.isLiveProcessing = false;
        this.liveStartTime = null;
        this.liveTimer = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.audioBuffer = [];
        this.chunkDuration = 15000; // 15 seconds in milliseconds
        this.overlapDuration = 3000; // 3 seconds overlap in milliseconds
        this.nextChunkTime = 0;
        this.websocket = null;
        this.liveQuestions = [];
        
        this.initializeEventListeners();
        this.checkApiHealth();
    }

    initializeEventListeners() {
        // Tab switching (commented out - only live processing now)
        // document.querySelectorAll('.tab-button').forEach(button => {
        //     button.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        // });

        // File upload (commented out)
        // const fileInput = document.getElementById('audioFileInput');
        // const uploadArea = document.getElementById('uploadArea');
        // 
        // fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        // 
        // uploadArea.addEventListener('click', () => fileInput.click());
        // uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        // uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        // uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Recording controls (commented out)
        // document.getElementById('recordButton').addEventListener('click', () => this.toggleRecording());
        // document.getElementById('stopButton').addEventListener('click', () => this.stopRecording());
        // document.getElementById('playButton').addEventListener('click', () => this.playRecording());

        // Process button (commented out)
        // document.getElementById('processButton').addEventListener('click', () => this.processAudio());

        // Live processing controls (main functionality)
        document.getElementById('liveStartButton').addEventListener('click', () => this.toggleLiveProcessing());
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // Reset states when switching tabs
        this.resetStates();
    }

    resetStates() {
        this.audioBlob = null;
        this.updateProcessButton();
        this.clearFileInfo();
        this.clearRecordingState();
        this.clearLiveProcessingState();
    }

    // File Upload Methods
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.validateAndSetFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.currentTarget.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.validateAndSetFile(files[0]);
        }
    }

    validateAndSetFile(file) {
        // Check file type
        if (!file.type.startsWith('audio/')) {
            this.showToast('Please select a valid audio file.', 'error');
            return;
        }

        // Check file size (50MB limit)
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            this.showToast('File size must be less than 50MB.', 'error');
            return;
        }

        this.audioBlob = file;
        this.displayFileInfo(file);
        this.updateProcessButton();
        this.showToast('Audio file selected successfully!', 'success');
    }

    displayFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');

        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
        fileInfo.style.display = 'block';
    }

    clearFileInfo() {
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('audioFileInput').value = '';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Recording Methods
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            this.recordingStartTime = Date.now();

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                this.audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.displayRecordingResult();
                this.updateProcessButton();
                
                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateRecordingUI();
            this.startRecordingTimer();
            
            this.showToast('Recording started!', 'info');

        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showToast('Unable to access microphone. Please check permissions.', 'error');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.stopRecordingTimer();
            this.updateRecordingUI();
            this.showToast('Recording stopped!', 'success');
        }
    }

    updateRecordingUI() {
        const recordButton = document.getElementById('recordButton');
        const recordingStatus = document.getElementById('recordingStatus');
        const recordingActions = document.getElementById('recordingActions');

        if (this.isRecording) {
            recordButton.classList.add('recording');
            recordButton.innerHTML = '<i class="fas fa-stop"></i> Recording...';
            recordingStatus.style.display = 'flex';
            recordingActions.style.display = 'none';
        } else {
            recordButton.classList.remove('recording');
            recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            recordingStatus.style.display = 'none';
            if (this.audioBlob) {
                recordingActions.style.display = 'flex';
            }
        }
    }

    startRecordingTimer() {
        this.recordingTimer = setInterval(() => {
            const elapsed = Date.now() - this.recordingStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            document.getElementById('recordingTimer').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    displayRecordingResult() {
        const recordedAudio = document.getElementById('recordedAudio');
        recordedAudio.src = URL.createObjectURL(this.audioBlob);
        recordedAudio.style.display = 'block';
    }

    playRecording() {
        const recordedAudio = document.getElementById('recordedAudio');
        if (recordedAudio.src) {
            recordedAudio.play();
        }
    }

    clearRecordingState() {
        this.isRecording = false;
        this.audioChunks = [];
        this.audioBlob = null;
        this.recordingStartTime = null;
        
        document.getElementById('recordingStatus').style.display = 'none';
        document.getElementById('recordingActions').style.display = 'none';
        document.getElementById('recordedAudio').style.display = 'none';
        document.getElementById('recordedAudio').src = '';
        
        this.updateRecordingUI();
        this.stopRecordingTimer();
    }

    // Live Processing Methods
    async toggleLiveProcessing() {
        if (this.isLiveProcessing) {
            await this.stopLiveProcessing();
        } else {
            await this.startLiveProcessing();
        }
    }

    async startLiveProcessing() {
        try {
            // Get microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000
                }
            });

            // Initialize Web Audio API
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create a ScriptProcessorNode for real-time audio processing
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            this.audioBuffer = [];
            this.nextChunkTime = Date.now() + this.chunkDuration;
            
            // Process audio data
            this.processor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                
                // Convert Float32Array to Int16Array for better compression
                const int16Data = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                }
                
                this.audioBuffer.push(...int16Data);
                
                // Check if it's time to send a chunk
                const currentTime = Date.now();
                if (currentTime >= this.nextChunkTime) {
                    this.processAudioChunk();
                    this.nextChunkTime = currentTime + this.chunkDuration - this.overlapDuration;
                }
            };
            
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            // Initialize WebSocket connection (commented out for now)
            // this.initializeWebSocket();
            
            // Start live processing
            this.isLiveProcessing = true;
            this.liveStartTime = Date.now();
            this.updateLiveProcessingUI();
            this.startLiveTimer();
            
            this.showToast('Live processing started!', 'success');
            
        } catch (error) {
            console.error('Error starting live processing:', error);
            if (error.name === 'NotAllowedError') {
                this.showToast('Microphone access denied. Please allow microphone permissions.', 'error');
            } else if (error.name === 'NotFoundError') {
                this.showToast('No microphone found. Please connect a microphone.', 'error');
            } else {
                this.showToast(`Unable to start live processing: ${error.message}`, 'error');
            }
        }
    }

    async stopLiveProcessing() {
        try {
            this.isLiveProcessing = false;
            
            // Stop audio processing
            if (this.processor) {
                this.processor.disconnect();
                this.processor = null;
            }
            
            if (this.audioContext) {
                await this.audioContext.close();
                this.audioContext = null;
            }
            
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
                this.mediaStream = null;
            }
            
            // Close Socket.IO connection
            if (this.websocket) {
                this.websocket.disconnect();
                this.websocket = null;
            }
            
            this.stopLiveTimer();
            this.updateLiveProcessingUI();
            
            this.showToast('Live processing stopped!', 'info');
            
        } catch (error) {
            console.error('Error stopping live processing:', error);
            this.showToast('Error stopping live processing', 'error');
        }
    }

    processAudioChunk() {
        if (this.audioBuffer.length === 0) return;
        
        // Ensure we have enough audio data (at least 1 second of audio)
        const minSamples = 16000; // 1 second at 16kHz
        if (this.audioBuffer.length < minSamples) {
            console.log('Not enough audio data yet, waiting...', this.audioBuffer.length);
            return;
        }
        
        console.log('Processing audio chunk with', this.audioBuffer.length, 'samples');
        
        // Create proper WAV file with headers
        const wavBlob = this.createWavBlob(this.audioBuffer, 16000);
        
        // Clear buffer (keep some overlap)
        const overlapSamples = Math.floor(this.overlapDuration * 16); // 16kHz sample rate
        this.audioBuffer = this.audioBuffer.slice(-overlapSamples);
        
        // Send audio chunk to backend
        this.sendAudioChunk(wavBlob);
    }

    createWavBlob(audioData, sampleRate) {
        const length = audioData.length;
        const buffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(buffer);
        
        // WAV file header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        // RIFF header
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        
        // fmt chunk
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // chunk size
        view.setUint16(20, 1, true); // audio format (PCM)
        view.setUint16(22, 1, true); // number of channels
        view.setUint32(24, sampleRate, true); // sample rate
        view.setUint32(28, sampleRate * 2, true); // byte rate
        view.setUint16(32, 2, true); // block align
        view.setUint16(34, 16, true); // bits per sample
        
        // data chunk
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // write audio data
        let offset = 44;
        for (let i = 0; i < length; i++) {
            view.setInt16(offset, audioData[i], true);
            offset += 2;
        }
        
        return new Blob([buffer], { type: 'audio/wav' });
    }

    async sendAudioChunk(audioBlob) {
        try {
            console.log('Sending audio chunk, size:', audioBlob.size, 'bytes');
            
            const formData = new FormData();
            formData.append('audio', audioBlob, `chunk_${Date.now()}.wav`);
            formData.append('timestamp', Date.now());
            
            const response = await fetch(`${this.apiBaseUrl}/process-audio`, {
                method: 'POST',
                body: formData,
                headers: {
                    'ngrok-skip-browser-warning': 'true'
                }
            });
            
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', errorText);
                this.showToast(`Server error: ${response.status}`, 'error');
                return;
            }
            
            const data = await response.json();
            console.log('Response data:', data);
            
            if (data.success) {
                // Log all metadata to console
                console.log('=== AUDIO PROCESSING METADATA ===');
                console.log('Transcription:', data.transcription);
                console.log('Language detection:', data.language_detection);
                console.log('Translation:', data.translation);
                console.log('Search Query:', data.semantic_search?.query_used);
                console.log('Search result:', data.semantic_search);
                console.log('Timestamp:', data.timestamp);
                console.log('==================================');
                
                this.displayLiveQuestion(data);
            } else {
                console.error('Processing failed:', data.error);
                this.showToast(`Processing failed: ${data.error}`, 'error');
            }
            
        } catch (error) {
            console.error('Error sending audio chunk:', error);
            this.showToast(`Network error: ${error.message}`, 'error');
        }
    }

    initializeWebSocket() {
        this.websocket = io(this.apiBaseUrl);
        
        this.websocket.on('connect', () => {
            console.log('Socket.IO connection established');
            this.updateConnectionStatus('Connected', 'success');
        });
        
        this.websocket.on('status', (data) => {
            console.log('Status update:', data.message);
        });
        
        this.websocket.on('question_result', (data) => {
            console.log('Received question result:', data);
            this.displayLiveQuestion(data);
        });
        
        this.websocket.on('disconnect', () => {
            console.log('Socket.IO connection closed');
            this.updateConnectionStatus('Disconnected', 'error');
        });
        
        this.websocket.on('connect_error', (error) => {
            console.error('Socket.IO connection error:', error);
            this.updateConnectionStatus('Connection Error', 'error');
        });
    }

    displayLiveQuestion(data) {
        console.log('Displaying live question:', data);
        
        // Log metadata to console for this live question
        console.log('=== LIVE QUESTION METADATA ===');
        console.log('User Transcription (not displayed):', data.transcription?.original_text);
        console.log('Language:', data.language_detection?.detected_language);
        console.log('Language Code:', data.language_detection?.language_code);
        console.log('Confidence:', data.language_detection?.confidence);
        console.log('Translation:', data.translation);
        console.log('Search Query Used:', data.semantic_search?.query_used);
        console.log('Database Questions & Answers (displayed):', data.semantic_search?.search_result?.results?.map(r => ({
            question: r.matched_question,
            answer: r.answer,
            score: r.similarity_score
        })));
        console.log('===============================');
        
        // Get the question from transcription (same as regular Record Audio tab)
        const transcribedQuestion = data.transcription?.original_text || '';
        
        // Get the search results (top 3 similar questions from embeddings)
        const searchResults = data.semantic_search?.search_result?.results || [];
        
        // Extract metadata for display  
        console.log('=== DEBUGGING TRANSLATION ===');
        console.log('Translation object:', data.translation);
        console.log('Translation type:', typeof data.translation);
        console.log('Translation keys:', data.translation ? Object.keys(data.translation) : 'null');
        console.log('=============================');
        
        // Extract translation text properly based on backend structure
        let translationText = 'No translation available';
        if (data.translation && data.translation.success) {
            translationText = data.translation.translated_text || 'Translation failed';
        } else if (data.translation && !data.translation.success) {
            translationText = `Translation failed: ${data.translation.error || 'Unknown error'}`;
        }
        
        const metadata = {
            detectedLanguage: data.language_detection?.detected_language || 'Unknown',
            punctuatedText: data.punctuation?.punctuated_text || transcribedQuestion,
            translation: translationText
        };
        
        const questionData = {
            timestamp: data.timestamp || Date.now(),
            question: transcribedQuestion,
            searchResults: searchResults,
            metadata: metadata
        };
        
        // Only add if we have actual question content
        if (questionData.question && questionData.question.trim().length > 3) {
            this.liveQuestions.unshift(questionData);
            
            // Keep only last 10 questions
            if (this.liveQuestions.length > 10) {
                this.liveQuestions = this.liveQuestions.slice(0, 10);
            }
            
            this.updateLiveQuestionsDisplay();
        } else {
            console.log('No sufficient question content to display');
        }
    }

    updateLiveQuestionsDisplay() {
        const container = document.getElementById('liveQuestionsList');
        const questionsContainer = document.getElementById('liveQuestionsContainer');
        
        if (this.liveQuestions.length === 0) {
            questionsContainer.style.display = 'none';
            return;
        }
        
        questionsContainer.style.display = 'block';
        
        container.innerHTML = this.liveQuestions.map((q, index) => `
            <div class="live-question-item ${index === 0 ? 'latest' : ''}">
                <div class="question-header">
                    <div class="question-timestamp">${this.formatTimestamp(q.timestamp)}</div>
                </div>
                
                <!-- Metadata (small font, secondary) -->
                <div class="question-metadata">
                    <div class="metadata-item">
                        <span class="metadata-label">Language:</span>
                        <span class="metadata-value">${q.metadata?.detectedLanguage || 'Unknown'}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Transcription:</span>
                        <span class="metadata-value">${q.metadata?.punctuatedText || q.question}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Translation:</span>
                        <span class="metadata-value">${String(q.metadata?.translation || 'No translation')}</span>
                    </div>
                </div>
                
                <!-- Multiple Clickable Questions -->
                ${q.searchResults.length > 0 ? `
                    ${q.searchResults.map((result, resultIndex) => `
                        <div class="clickable-question" onclick="app.toggleQuestionAnswer(${index}, ${resultIndex})">
                            <div class="question-text">${result.matched_question}</div>
                            <div class="question-toggle">
                                <i class="fas fa-chevron-down"></i>
                            </div>
                        </div>
                        
                        <!-- Answer (hidden by default) -->
                        <div class="question-answers" id="answers-${index}-${resultIndex}" style="display: none;">
                            <div class="single-answer">
                                <div class="result-answer">${result.answer}</div>
                            </div>
                        </div>
                    `).join('')}
                ` : `
                    <div class="no-results">
                        <p>No matching questions found in database</p>
                    </div>
                `}
            </div>
        `).join('');
    }

    updateLiveProcessingUI() {
        const startButton = document.getElementById('liveStartButton');
        const liveStatus = document.getElementById('liveStatus');
        const connectionStatus = document.getElementById('connectionStatus');
        
        if (this.isLiveProcessing) {
            startButton.classList.add('processing');
            startButton.innerHTML = '<i class="fas fa-stop"></i> Stop Live Processing';
            liveStatus.style.display = 'flex';
            connectionStatus.style.display = 'flex';
        } else {
            startButton.classList.remove('processing');
            startButton.innerHTML = '<i class="fas fa-play"></i> Start Live Processing';
            liveStatus.style.display = 'none';
            connectionStatus.style.display = 'none';
        }
    }

    startLiveTimer() {
        this.liveTimer = setInterval(() => {
            const elapsed = Date.now() - this.liveStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            document.getElementById('liveTimer').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    stopLiveTimer() {
        if (this.liveTimer) {
            clearInterval(this.liveTimer);
            this.liveTimer = null;
        }
    }

    updateConnectionStatus(status, type) {
        const connectionText = document.getElementById('connectionText');
        const statusIndicator = document.querySelector('.status-indicator');
        
        connectionText.textContent = status;
        statusIndicator.className = `status-indicator ${type}`;
    }

    formatTimestamp(timestamp) {
        try {
            // Handle both string and number timestamps
            const ts = typeof timestamp === 'string' ? parseInt(timestamp) : timestamp;
            const date = new Date(ts);
            
            // Check if date is valid
            if (isNaN(date.getTime())) {
                return new Date().toLocaleTimeString(); // Fallback to current time
            }
            
            return date.toLocaleTimeString();
        } catch (error) {
            console.error('Error formatting timestamp:', error, 'timestamp:', timestamp);
            return new Date().toLocaleTimeString(); // Fallback to current time
        }
    }

    clearLiveProcessingState() {
        this.isLiveProcessing = false;
        this.liveQuestions = [];
        this.audioBuffer = [];
        this.nextChunkTime = 0;
        
        document.getElementById('liveQuestionsContainer').style.display = 'none';
        document.getElementById('liveQuestionsList').innerHTML = '';
        document.getElementById('connectionStatus').style.display = 'none';
        
        this.updateLiveProcessingUI();
        this.stopLiveTimer();
    }

    toggleQuestionAnswer(questionIndex, resultIndex) {
        const answersElement = document.getElementById(`answers-${questionIndex}-${resultIndex}`);
        const questionElement = answersElement.previousElementSibling;
        const toggleIcon = questionElement.querySelector('.question-toggle i');
        
        if (answersElement.style.display === 'none') {
            answersElement.style.display = 'block';
            toggleIcon.className = 'fas fa-chevron-up';
        } else {
            answersElement.style.display = 'none';
            toggleIcon.className = 'fas fa-chevron-down';
        }
    }

    // Process Button
    updateProcessButton() {
        const processButton = document.getElementById('processButton');
        processButton.disabled = !this.audioBlob;
    }

    // API Methods
    async checkApiHealth() {
        try {
            console.log('Checking API health at:', this.apiBaseUrl);
            const response = await fetch(`${this.apiBaseUrl}/health`, {
                headers: {
                    'ngrok-skip-browser-warning': 'true'
                }
            });
            
            console.log('Response status:', response.status);
            console.log('Response ok:', response.ok);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('API response:', data);
            
            if (data.status === 'healthy') {
                this.showToast('Connected to API successfully!', 'success');
            } else {
                this.showToast(`API status: ${data.status}`, 'warning');
            }
        } catch (error) {
            console.error('API health check failed:', error);
            console.error('API URL being used:', this.apiBaseUrl);
            console.error('Error details:', error.message);
            
            // More specific error messages
            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                this.showToast(`Cannot reach API at ${this.apiBaseUrl}. Check if backend is running and URL is correct.`, 'error');
            } else if (error.message.includes('CORS')) {
                this.showToast('CORS error: Backend may not allow requests from this domain.', 'error');
            } else {
                this.showToast(`API Error: ${error.message}`, 'error');
            }
        }
    }

    async processAudio() {
        if (!this.audioBlob) {
            this.showToast('Please select or record an audio file first.', 'warning');
            return;
        }

        this.showLoadingState();
        
        try {
            const formData = new FormData();
            formData.append('audio', this.audioBlob, 'audio.wav');

            const response = await fetch(`${this.apiBaseUrl}/process-audio`, {
                method: 'POST',
                body: formData,
                headers: {
                    'ngrok-skip-browser-warning': 'true'
                }
            });

            const data = await response.json();

            if (data.success) {
                this.displayResults(data);
                this.showToast('Audio processed successfully!', 'success');
            } else {
                throw new Error(data.error || 'Processing failed');
            }

        } catch (error) {
            console.error('Error processing audio:', error);
            this.showToast(`Error: ${error.message}`, 'error');
            this.hideLoadingState();
        }
    }

    showLoadingState() {
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('loadingState').style.display = 'block';
        document.getElementById('resultsContent').style.display = 'none';
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    hideLoadingState() {
        document.getElementById('loadingState').style.display = 'none';
    }

    displayResults(data) {
        this.hideLoadingState();
        
        // Log all metadata to console
        console.log('=== AUDIO PROCESSING RESULTS ===');
        console.log('Language Detection:', data.language_detection);
        console.log('Transcription:', data.transcription);
        console.log('Translation:', data.translation);
        console.log('Search Query:', data.semantic_search?.query_used);
        console.log('Search Results:', data.semantic_search);
        console.log('================================');
        
        // Display only question and answer
        this.displayQuestionAndAnswer(data);
        
        document.getElementById('resultsContent').style.display = 'block';
    }

    displayQuestionAndAnswer(data) {
        // Display search results with database questions and answers
        const searchResults = document.getElementById('searchResults');
        const result = data.semantic_search?.search_result;
        
        if (result?.status === 'success' && result.results && result.results.length > 0) {
            // Show the first result's question as the main question
            document.getElementById('questionText').textContent = result.results[0].matched_question;
            
            // Show all answers
            searchResults.innerHTML = result.results.map((item, index) => `
                <div class="search-result-item">
                    <div class="result-answer">${item.answer}</div>
                </div>
            `).join('');
        } else if (result?.status === 'no_match') {
            document.getElementById('questionText').textContent = 'No matching question found';
            searchResults.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-search"></i>
                    <h3>No exact matches found</h3>
                    <p>Try rephrasing your question or using different keywords.</p>
                </div>
            `;
        } else {
            document.getElementById('questionText').textContent = 'Search failed';
            searchResults.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Search failed</h3>
                    <p>Unable to find relevant answers. Please try again.</p>
                </div>
            `;
        }
    }

    // Utility Methods
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        toast.innerHTML = `
            <i class="toast-icon ${icons[type]}"></i>
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }

    clearResults() {
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'none';
        document.getElementById('loadingState').style.display = 'none';
    }
}

// Global functions for HTML onclick handlers
function removeFile() {
    app.clearFileInfo();
    app.audioBlob = null;
    app.updateProcessButton();
}

function clearResults() {
    app.clearResults();
}

// Initialize the application when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new VoiceRecommendationApp();
});

// Handle page visibility changes to stop recording if user switches tabs
document.addEventListener('visibilitychange', () => {
    if (document.hidden && app) {
        if (app.isRecording) {
            app.stopRecording();
            app.showToast('Recording stopped due to tab switch', 'warning');
        }
        if (app.isLiveProcessing) {
            app.stopLiveProcessing();
            app.showToast('Live processing stopped due to tab switch', 'warning');
        }
    }
});

// Handle beforeunload to clean up resources
window.addEventListener('beforeunload', () => {
    if (app) {
        if (app.isRecording) {
            app.stopRecording();
        }
        if (app.isLiveProcessing) {
            app.stopLiveProcessing();
        }
    }
});
