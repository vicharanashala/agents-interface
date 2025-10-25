#!/usr/bin/env node

/**
 * Setup script for configuring ngrok URLs
 * Run this script to automatically update the config.js with your ngrok URLs
 */

const fs = require('fs');
const readline = require('readline');

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function setupNgrok() {
    console.log('\n🔧 KCC\'s Friend - Ngrok Setup');
    console.log('================================\n');
    
    console.log('This script will help you configure ngrok URLs for your Voice Recommendation System.\n');
    
    console.log('📋 Instructions:');
    console.log('1. Make sure your backend is running on localhost:5000');
    console.log('2. In a separate terminal, run: ngrok http 5000');
    console.log('3. Copy the HTTPS URL from ngrok (e.g., https://abc123.ngrok-free.app)');
    console.log('4. Enter that URL below\n');
    
    const backendUrl = await askQuestion('Enter your ngrok backend URL (https://...ngrok-free.app): ');
    
    if (!backendUrl.startsWith('https://') || !backendUrl.includes('ngrok')) {
        console.log('❌ Invalid ngrok URL. Please make sure it starts with https:// and contains ngrok');
        rl.close();
        return;
    }
    
    // Update config.js
    try {
        let configContent = fs.readFileSync('config.js', 'utf8');
        
        // Replace the NGROK_BACKEND URL
        configContent = configContent.replace(
            /NGROK_BACKEND: '[^']*'/,
            `NGROK_BACKEND: '${backendUrl}'`
        );
        
        fs.writeFileSync('config.js', configContent);
        
        console.log('\n✅ Configuration updated successfully!');
        console.log(`🔗 Backend URL: ${backendUrl}`);
        console.log(`🔗 Frontend URL: https://84923a4b919d.ngrok-free.app`);
        
        console.log('\n📝 Next steps:');
        console.log('1. Make sure your backend ngrok is still running');
        console.log('2. Open your frontend ngrok URL in a browser');
        console.log('3. Test the voice recommendation system');
        
        console.log('\n🧪 Test your setup:');
        console.log(`curl -s ${backendUrl}/health`);
        
    } catch (error) {
        console.log('❌ Error updating configuration:', error.message);
    }
    
    rl.close();
}

function askQuestion(question) {
    return new Promise((resolve) => {
        rl.question(question, (answer) => {
            resolve(answer.trim());
        });
    });
}

// Run the setup if this script is executed directly
if (require.main === module) {
    setupNgrok().catch(console.error);
}

module.exports = { setupNgrok };
