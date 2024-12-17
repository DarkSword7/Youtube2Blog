import os
import re
import warnings
import nltk
import whisper
import tempfile
import pytube
import yt_dlp
from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
import google.generativeai as genai
import moviepy.editor as mp

# Suppress pytube warnings
warnings.filterwarnings('ignore')

# Ensure necessary downloads
nltk.download('punkt', quiet=True)

# Configure Google Gemini API
GOOGLE_API_KEY = "AIzaSyCC_58lj4F6wwEDEC6H4B2ppYNTylc36LE"  # Replace with your actual API key
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)

# Load Whisper model (select model size based on your system capabilities)
# Options: 'tiny', 'base', 'small', 'medium', 'large'
# Smaller models are faster but less accurate
WHISPER_MODEL = 'small'  # Adjust based on your computational resources

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
        Generate a blog post using Google Gemini AI.
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

        # Prepare input for Gemini AI
        input_text = f"""
        You are an expert blog post writer.

        Video Details:
        - Title: {video_details.get('title', 'Unknown Title')}
        - Description: {video_details.get('description', 'No description available')}
        - Author: {video_details.get('author', 'Unknown Creator')}
        - Length: {video_details.get('length', 'Unknown')} seconds
        - Views: {video_details.get('views', 'Not available')}

        Transcript Content:
        {transcript}

        Task:
        Based on the video details and transcript, write a detailed and well-structured blog post. The blog post should include:
        1. A captivating title (without any hashtags or unnecessary formatting).
        2. A compelling introduction that introduces the topic.
        3. Organized main content with clear headings, bullet points, and insights.
        4. A thoughtful conclusion that summarizes key points and adds value.

        **Output the blog in plain text format without any markdown, extra hashtags, or formatting issues.**
        """
        # Use Gemini AI to generate the blog post
        model = genai.GenerativeModel('gemini-pro')
        try:
            response = model.generate_content(input_text)
            
            # Debugging: Log the raw response from AI
            print("AI Raw Response:", response.text)

            # Validate AI response
            if not response.text.strip() or "Introduction" in response.text and "Main Content" in response.text:
                raise ValueError("AI response is incomplete or contains placeholders.")
            
            # Clean and parse blog post content
            blog_post = self._clean_and_format_blog_post(response.text)

            return blog_post
        except Exception as e:
            return {"error": f"AI generation failed: {str(e)}"}

    def _clean_and_format_blog_post(self, blog_post):
        """
        Clean and format the AI-generated blog post:
        - Remove redundant introductions/conclusions
        - Remove hashtags from headings
        - Improve formatting
        
        :param blog_post: Raw AI-generated blog content
        :return: Formatted blog content as a dictionary
        """
        import re

        # Split blog into sections
        sections = re.split(r'\n\n+', blog_post)
        clean_sections = []
        seen_titles = set()

        for section in sections:
            # Remove headings with hashtags (## Heading -> Heading)
            section = re.sub(r'^\s*#+\s*', '', section.strip())

            # Remove duplicate titles (e.g., repeated "Introduction", "Conclusion")
            if section in seen_titles:
                continue
            seen_titles.add(section)

            clean_sections.append(section)

        # Combine clean sections back into content
        formatted_blog = "\n\n".join(clean_sections)

        # Extract structured content
        return {
            "title": clean_sections[0] if clean_sections else "Generated Blog Post",
            "content": formatted_blog
        }
# Flask Application Setup
app = Flask(__name__)

@app.route('/')
def home():
    """
    Render the home page
    """
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate blog post from YouTube video URL
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
        
        return jsonify({
            "title": blog_post["title"],
            "content": blog_post["content"]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

# Requirements to install:
# pip install flask youtube_transcript_api google-generativeai pytube nltk 
# pip install openai-whisper --upgrade
# pip install moviepy
# pip install torch
# 
# For Whisper, you might also need to:
# pip install setuptools-rust
# 
# Note: Whisper model download happens automatically on first use