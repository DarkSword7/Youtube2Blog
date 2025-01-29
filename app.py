import os
import re
import warnings
import nltk
import whisper
import tempfile
import pytube
import yt_dlp
from flask import Flask, render_template, request, jsonify, redirect, url_for
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
from groq import Groq
import moviepy.editor as mp
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import requests
import time

# Suppress pytube warnings
warnings.filterwarnings('ignore')

# Ensure necessary downloads
nltk.download('punkt', quiet=True)

# Load Whisper model (select model size based on your system capabilities)
# Options: 'tiny', 'base', 'small', 'medium', 'large'
# Smaller models are faster but less accurate
WHISPER_MODEL = 'small'  # Adjust based on your computational resources

# Configure Groq client
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Add a test function to verify API connection
def test_api_connection():
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
        )
        return True
    except Exception as e:
        print(f"API test failed: {str(e)}")
        return False

class YouTubeToBlogConverter:
    def __init__(self, video_url, target_language='en'):
        """
        Initialize the YouTube to Blog converter with enhanced capabilities
        
        :param video_url: URL of the YouTube video
        :param target_language: Language for translation (default: English)
        """
        self.video_url = video_url
        self.target_language = target_language
        self.video_id = self._extract_video_id()
        self.transcript = None
        self.video_details = None
        self.whisper_model = None

    def _extract_video_id(self):
        """
        Extract YouTube video ID from the given URL
        
        :return: YouTube video ID
        """
        # Previous implementation remains the same as in the original script
        normalized_url = self.video_url.strip().replace(' ', '')
        
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?&]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?&]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([^?&]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, normalized_url)
            if match:
                return match.group(1)
        
        parsed_url = urlparse(normalized_url)
        
        if 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.strip('/')
        
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v', [None])[0]
        
        if video_id:
            return video_id
        
        raise ValueError(f"Unable to extract video ID from URL: {self.video_url}")

    def _download_video(self):
        """
        Download YouTube video to a temporary file
        
        :return: Path to the downloaded video file
        """
        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Download the highest resolution audio stream
            youtube_video = pytube.YouTube(self.video_url)
            audio_stream = youtube_video.streams.filter(only_audio=True).first()
            
            # Download the audio
            audio_file_path = audio_stream.download(output_path=temp_dir)
            
            return audio_file_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            raise

    def _extract_audio(self, video_path):
        """
        Extract audio from video file
        
        :param video_path: Path to the video file
        :return: Path to the extracted audio file
        """
        try:
            # Create a temporary directory for audio
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, 'audio.wav')
            
            # Extract audio using moviepy
            video_clip = mp.AudioFileClip(video_path)
            video_clip.write_audiofile(audio_path)
            video_clip.close()
            
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise

    def _transcribe_audio(self, audio_path):
        """
        Transcribe audio using Whisper
        
        :param audio_path: Path to the audio file
        :return: Transcribed text
        """
        try:
            # Lazy load Whisper model to save memory
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model(WHISPER_MODEL)
            
            # Transcribe the audio
            result = self.whisper_model.transcribe(audio_path)
            
            return result['text']
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise

    def fetch_video_transcript(self):
        """
        Attempt to fetch transcript. If not available, use speech-to-text
        
        :return: Transcript text
        """
        try:
            # First, try to get YouTube's transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
            transcript = transcript_list.find_generated_transcript([self.target_language])
            self.transcript = ' '.join([entry['text'] for entry in transcript.fetch()])
            return self.transcript
        except (TranscriptsDisabled, NoTranscriptFound):
            # If no transcript, use speech-to-text
            print("No transcript found. Attempting speech-to-text...")
            
            try:
                # Download video
                video_path = self._download_video()
                
                # Extract audio
                audio_path = self._extract_audio(video_path)
                
                # Transcribe audio
                self.transcript = self._transcribe_audio(audio_path)
                
                return self.transcript
            except Exception as e:
                print(f"Failed to generate transcript: {e}")
                return None

    def fetch_video_details(self):
        try:
            ydl_opts = {'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.video_url, download=False)
                self.video_details = {
                    'title': info.get('title', 'Unknown Title'),
                    'description': info.get('description', 'No description available'),
                    'author': info.get('uploader', 'Unknown Creator'),
                    'length': info.get('duration', 0),
                    'views': info.get('view_count', 0),
                    'rating': info.get('average_rating', 0)
                }
                return self.video_details
        except Exception as e:
            print(f"Error fetching video details with yt_dlp: {e}")
            return {
                'title': 'Unknown Title',
                'description': 'No description available',
                'author': 'Unknown Creator',
                'length': 0,
                'views': 0,
                'rating': 0
            }

    def generate_blog_with_ai(self):
        """
        Generate a blog post using Groq's Llama 3.3 70B model.
        Ensures the response is validated and structured correctly.
        """
        print("Attempting to fetch transcript...")
        transcript = self.fetch_video_transcript()

        print("Fetching video details...")
        try:
            video_details = self.fetch_video_details()
            print("Video details:", video_details)
        except Exception as e:
            print(f"Error in video details: {e}")
            video_details = {}

        if not transcript:
            return {"error": "Could not generate transcript for this video."}

        try:
            # Test API connection first
            if not test_api_connection():
                raise ValueError("Failed to connect to the API. Please check your API key and connection.")

            print("Making direct API request...")
            prompt = f"""Create a blog post about this video:
            Video Information:
            - Title: {video_details.get('title', 'Unknown Title')}
            - Author: {video_details.get('author', 'Unknown Creator')}
            
            Instructions:
            1. Create a CONCISE title (maximum 60 characters) that captures the main topic
            2. Title should be clear, engaging, and straight to the point
            3. Avoid questions in the title
            4. Don't include "Introduction" or similar prefix words
            
            Transcript Summary:
            {transcript[:2000]}...
            
            Format as a markdown blog post with:
            1. The concise title as a single H1 heading
            2. Well-organized sections with H2 headings
            3. Clear and engaging content
            4. Proper markdown formatting"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert blog writer who specializes in creating engaging content with concise, impactful titles.
                        Key rules for titles:
                        - Keep titles under 60 characters
                        - Be specific and direct
                        - Avoid questions or vague statements
                        - Focus on the main value or insight
                        - Don't use unnecessary prefixes like 'Introduction to' or 'Guide to'"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
            )

            if not response:
                raise ValueError("No response received from API")

            print("API Response received")
            print("Response type:", type(response))
            print("Response attributes:", dir(response))

            if not hasattr(response, 'choices'):
                raise ValueError("Invalid response format from API")

            response_text = response.choices[0].message.content
            print("Generated content:", response_text[:200] + "...")

            # Clean and parse blog post content
            blog_post = self._clean_and_format_blog_post(response_text)
            
            if not blog_post.get("title") or not blog_post.get("content"):
                raise ValueError("Generated blog post is missing title or content")

            return blog_post

        except Exception as e:
            error_msg = f"Error generating blog post: {str(e)}"
            print(error_msg)
            print("Full error details:", e)
            print("Response received:", locals().get('response', 'No response'))
            return {"error": error_msg}

    def _clean_and_format_blog_post(self, blog_post):
        """
        Clean and format the AI-generated blog post
        """
        try:
            import re

            # Split blog into sections
            sections = re.split(r'\n\n+', blog_post)
            clean_sections = []
            seen_titles = set()

            # Ensure we have content to process
            if not sections:
                raise ValueError("No content sections found in AI response")

            for section in sections:
                # Remove headings with hashtags (## Heading -> Heading)
                section = re.sub(r'^\s*#+\s*', '', section.strip())

                # Remove duplicate titles
                if section in seen_titles:
                    continue
                seen_titles.add(section)

                clean_sections.append(section)

            # Combine clean sections back into content
            formatted_blog = "\n\n".join(clean_sections)

            # Extract title (look for "Title:" prefix or use first section)
            title = None
            for section in clean_sections:
                if section.lower().startswith("title:"):
                    title = section.replace("Title:", "").strip()
                    break
            
            if not title:
                title = clean_sections[0] if clean_sections else "Generated Blog Post"

            return {
                "title": title,
                "content": formatted_blog
            }
        except Exception as e:
            print(f"Error in _clean_and_format_blog_post: {str(e)}")
            raise

# Flask Application Setup
app = Flask(__name__)

# Add these configurations after app initialization
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Add this after the User model
class Blog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_url = db.Column(db.String(500), nullable=False)
    language = db.Column(db.String(10), nullable=False)

    user = db.relationship('User', backref=db.backref('blogs', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    """
    Render the home page
    """
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400

        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()

        login_user(user)
        return jsonify({'message': 'Registration successful'})

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return jsonify({'message': 'Login successful'})
        
        return jsonify({'error': 'Invalid username or password'}), 401

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/generate', methods=['POST'])
@login_required
def generate():
    """
    Generate blog post from YouTube video URL and save it
    """
    video_url = request.form.get('video_url')
    target_language = request.form.get('target_language', 'en')

    if not video_url:
        return jsonify({'error': 'Video URL is required'}), 400

    try:
        converter = YouTubeToBlogConverter(video_url, target_language)
        blog_post = converter.generate_blog_with_ai()
        
        if 'error' in blog_post:
            return jsonify(blog_post), 400
        
        # Save the blog
        new_blog = Blog(
            title=blog_post["title"],
            content=blog_post["content"],
            user_id=current_user.id,
            video_url=video_url,
            language=target_language
        )
        db.session.add(new_blog)
        db.session.commit()
        
        return jsonify({
            "title": blog_post["title"],
            "content": blog_post["content"],
            "id": new_blog.id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add route to get user's blogs
@app.route('/blogs')
@login_required
def get_blogs():
    blogs = Blog.query.filter_by(user_id=current_user.id).order_by(Blog.created_at.desc()).all()
    return jsonify([{
        'id': blog.id,
        'title': blog.title,
        'created_at': blog.created_at.strftime('%Y-%m-%d %H:%M'),
        'language': blog.language
    } for blog in blogs])

# Add route to get a specific blog
@app.route('/blog/<int:blog_id>')
@login_required
def get_blog(blog_id):
    blog = Blog.query.get_or_404(blog_id)
    if blog.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({
        'id': blog.id,
        'title': blog.title,
        'content': blog.content,
        'video_url': blog.video_url,
        'language': blog.language,
        'created_at': blog.created_at.strftime('%Y-%m-%d %H:%M')
    })

@app.route('/blog/<int:blog_id>', methods=['DELETE'])
@login_required
def delete_blog(blog_id):
    blog = Blog.query.get_or_404(blog_id)
    if blog.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        db.session.delete(blog)
        db.session.commit()
        return jsonify({'message': 'Blog deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Add this after app initialization but before route definitions
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run()

# Requirements to install:
# pip install flask youtube_transcript_api groq pytube nltk 
# pip install openai-whisper --upgrade
# pip install moviepy
# pip install torch
# 
# For Whisper, you might also need to:
# pip install setuptools-rust
# 
# Note: Whisper model download happens automatically on first use

# Additional requirements:
# pip install flask-sqlalchemy flask-login