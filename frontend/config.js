// Configuration for Voice Recommendation System Frontend
const CONFIG = {
    // API Configuration
    API: {
        // Local development
        LOCAL: 'http://localhost:5020',
        
        // Ngrok URLs (update these with your actual ngrok URLs)
        NGROK_BACKEND: 'https://hyperlogical-soppiest-krystin.ngrok-free.dev/api/backend/',
        
        // Tailscale URL (will be auto-detected)
        TAILSCALE_BACKEND: '', // Auto-detected from hostname
        
        // Auto-detect environment
        get BASE_URL() {
            const hostname = window.location.hostname;
            
            // Check if we're running on Tailscale (hostname contains .ts.net)
            if (hostname.includes('.ts.net')) {
                // If frontend is on a subdomain, backend is on main domain
                if (hostname.startsWith('frontend.')) {
                    const backendHost = hostname.replace('frontend.', '');
                    return `https://${backendHost}`;
                }
                // If frontend is on main domain, backend is also on main domain
                return `https://${hostname}`;
            }
            
            // Check if we're running on ngrok (frontend URL contains ngrok)
            if (hostname.includes('ngrok') || hostname.includes('ngrok-free.app')) {
                return this.NGROK_BACKEND;
            }
            
            // Default to local for development
            return this.LOCAL;
        }
    },
    
    // Audio processing settings
    AUDIO: {
        MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
        CHUNK_DURATION: 15000, // 15 seconds
        OVERLAP_DURATION: 3000, // 3 seconds
        SAMPLE_RATE: 16000
    },
    
    // UI settings
    UI: {
        MAX_LIVE_QUESTIONS: 10,
        TOAST_DURATION: 5000,
        SUPPORTED_FORMATS: [
            'audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac',
            'audio/ogg', 'audio/m4a', 'audio/aac', 'audio/mp4',
            'audio/wma', 'audio/amr', 'audio/aiff', 'audio/au',
            'audio/3gpp', 'audio/webm'
        ]
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
} else if (typeof window !== 'undefined') {
    window.CONFIG = CONFIG;
}
