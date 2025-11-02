import os
from textwrap import dedent
from fastapi import FastAPI
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app
from agno.app.playground.settings import PlaygroundSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
openrouter_key = os.getenv('OPENROUTER_API_KEY')

if not openrouter_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found")
    exit(1)

print("✅ OpenRouter API key loaded successfully!")
print("🎬 Starting Movie Recommendation Agent with DeepSeek via OpenRouter...")

# ✅ Get your public URL for better logging
public_url = os.getenv('RAILWAY_STATIC_URL', 'movie-recommendation-agent-production.up.railway.app')
port = int(os.getenv('PORT', 7777))

print(f"🌐 Public URL: {public_url}")
print(f"🔧 Port: {port}")

# Create agent
movie_recommendation_agent = Agent(
    name='Movie Recommendation Agent',
    model=OpenAIChat(
        id='deepseek/deepseek-chat',
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=2000
    ),
    tools=[],
    description=dedent("""\
        You are a **passionate and knowledgeable movie expert**. Your mission is to help users **discover their next favorite movies** by providing **detailed, personalized, and exciting recommendations**.

        ### **Your Approach**
        - Analyze user input to **understand their tastes, favorite genres, and specific preferences**.
        - Curate recommendations using a mix of **classic masterpieces, hidden gems, and trending films**.
        - Ensure each suggestion is **relevant, diverse, and backed by strong ratings and reviews**.
        - Provide **up-to-date information** on movie details, including cast, director, runtime, and content advisories**.
        - Highlight **where to watch**, suggest **upcoming releases**, and include **trailers when available**.

        ### **Your Recommendations Should Include:**
        - **Title & Release Year**
        - **Genre & Subgenres** (with emoji indicators)
        - **IMDb Rating** (Focus on 7.5+ rated films)
        - **Runtime & Primary Language**
        - **Engaging Plot Summary**
        - **Content Advisory / Age Rating**
        - **Notable Cast & Director**

        ### **Presentation Guidelines**
        - Use **clear Markdown formatting** for readability.
        - Organize recommendations in a **structured table**.
        - **Group similar movies together** for better discovery.
        - Provide **at least 5 personalized recommendations per query**.
        - Offer a **brief explanation** for why each movie was selected.
    """),
    instructions=dedent("""\
        ### Approach for Generating Recommendations
        
        #### 1. **Analysis Phase**
        - Interpret user preferences based on input.
        - Analyze favorite movies for themes, styles, and patterns.
        - Consider specific user requirements (e.g., genre, rating, language, mood).

        #### 2. **Search & Curation**
        - Use your extensive movie knowledge to find relevant options.
        - Ensure variety in recommendations (mix of classics, hidden gems, and trending titles).
        - Verify that movie details are accurate based on your training data.

        #### 3. **Detailed Information for Each Recommendation**
        Each movie recommendation should include:
        - Title & Release Year
        - Genre & Subgenres
        - IMDb Rating (Focus on 7.5+ rated films)
        - Runtime & Primary Language
        - Brief, Engaging Plot Summary
        - Content Advisory / Age Rating
        - Notable Cast & Director

        #### 4. **Additional Features**
        - Include streaming availability when possible.
        - Suggest similar movies in related genres.
        - Mention why each movie suits the user's request.

        #### **Presentation Style**
        - Format output using clear Markdown structure.
        - Present main recommendations in a structured table.
        - Group similar movies together for easy browsing.
        - Use emoji indicators to visually represent genres (e.g., 🎭 Drama, 💥 Action, 🗺️ Adventure).
        - Provide a minimum of 5 recommendations per query.
        - Offer a brief explanation of why each movie was recommended.
    """),
    markdown=True,
)

# ✅ Create FastAPI app FIRST
app = FastAPI(
    title="Movie Recommendation API",
    description="AI-powered movie recommendation system using DeepSeek via OpenRouter",
    version="1.0.0"
)

# ✅ Add your custom routes to the FastAPI app
@app.get("/")
async def root():
    return {
        "message": "Movie Recommendation API", 
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "info": "/info",
            "playground": "/playground"
        },
        "agent": "Movie Recommendation Agent with DeepSeek",
        "model": "deepseek/deepseek-chat",
        "public_url": f"https://{public_url}"
    }

@app.get("/health")
async def health_check():
    import datetime
    return {
        "status": "healthy", 
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "service": "Movie Recommendation API",
        "public_url": f"https://{public_url}"
    }

@app.get("/info")
async def api_info():
    return {
        "name": "Movie Recommendation API",
        "version": "1.0.0",
        "description": "AI-powered movie recommendation system",
        "public_url": f"https://{public_url}",
        "features": [
            "Personalized movie recommendations",
            "DeepSeek AI integration via OpenRouter",
            "Interactive playground interface",
            "RESTful API endpoints"
        ],
        "playground_url": f"https://{public_url}/playground"
    }

# ✅ FIXED: Use only valid PlaygroundSettings parameters
settings = PlaygroundSettings(env="dev")

# ✅ Create and mount the playground
playground_app = Playground(
    agents=[movie_recommendation_agent], 
    settings=settings
).get_app()

app.mount("/playground", playground_app)

# ✅ Updated main block with better logging
if __name__ == '__main__':
    print(f"🌐 Starting server on port {port}")
    print(f"🚀 Public URL: https://{public_url}")
    print(f"📚 API Documentation: https://{public_url}/docs")
    print(f"🎮 Playground: https://{public_url}/playground")
    print(f"❤️  Health Check: https://{public_url}/health")
    print(f"🔍 API Info: https://{public_url}/info")
    
    serve_playground_app('agentMovieRecommendation:app', host='0.0.0.0', port=port, reload=False)