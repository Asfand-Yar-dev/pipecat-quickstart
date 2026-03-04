import os
from dotenv import load_dotenv
from loguru import logger
import numpy as np

# PDF / DOCX parsing
import fitz        # pip install PyMuPDF
from docx import Document  # pip install python-docx

# Sentence embeddings for RAG
from sentence_transformers import SentenceTransformer

# Pipecat imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregatorParams, LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
import pipecat.services.deepgram.stt

logger.info("🚀 Starting Pipecat + RAG bot...")

# -------------------
# Load environment variables
# -------------------
load_dotenv(override=True)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# -------------------
# System prompt
# -------------------
SYSTEM_PROMPT = "You are an intelligent assistant. Answer questions ONLY based on the provided document context."

# -------------------
# Load any document (PDF / DOCX)
# -------------------
def load_document_text(file_path):
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError("Unsupported format! Use PDF or DOCX.")

doc_file = "Resume_BD.pdf"  # Change this to your file
doc_text = load_document_text(doc_file)

# -------------------
# Chunk & embed text for RAG
# -------------------
def chunk_text(text, chunk_size=250):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

doc_chunks = chunk_text(doc_text)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(doc_chunks)

def retrieve_relevant(query, chunks, embeddings, top_k=3):
    query_vec = embedder.encode([query])[0]
    # Normalize embeddings to avoid NaN
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_vec_norm = query_vec / np.linalg.norm(query_vec)
    sims = np.dot(embeddings_norm, query_vec_norm)
    top_idx = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx]

# -------------------
# Aggregator to combine frames
# -------------------
class FullResponseAggregator(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aggregation = ""
        self._collecting = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._aggregation = ""
            self._collecting = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, (TextFrame, LLMTextFrame)) and self._collecting:
            self._aggregation += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._collecting = False
            if self._aggregation.strip():
                await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame, direction)
            self._aggregation = ""
        else:
            await self.push_frame(frame, direction)

# -------------------
# Run bot with RAG
# -------------------
async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):

    stt = pipecat.services.deepgram.stt.DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
    tts = CartesiaTTSService(api_key=CARTESIA_API_KEY, voice_id="6ccbfb76-1fc6-48f7-b71d-91ac6298247b")
    llm = OpenAILLMService(
        api_key="not-needed",
        base_url=LM_STUDIO_BASE_URL,
        model=MODEL_NAME,
        params=OpenAILLMService.InputParams(max_completion_tokens=512, temperature=0.2)
    )

    # Conversation context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        assistant_params=LLMAssistantAggregatorParams(expect_stripped_words=True)
    )

    # Pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        FullResponseAggregator(),
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True))

    # -------------------
    # Inject RAG
    # -------------------
    original_process_frame = user_aggregator.process_frame

    async def rag_process_frame(frame, direction):
        from pipecat.frames.frames import TextFrame
        if isinstance(frame, TextFrame):
            user_query = frame.text.strip()
            if not user_query:
                await original_process_frame(frame, direction)
                return

            relevant_chunks = retrieve_relevant(user_query, doc_chunks, doc_embeddings, top_k=3)
            context_text = "\n".join([f"- {chunk}" for chunk in relevant_chunks])

            # Merge system + RAG context
            if "interview question" in user_query.lower() or "questions from resume" in user_query.lower():
                frame.text = (
                    f"{SYSTEM_PROMPT}\n\n[DOCUMENT CONTEXT]:\n{context_text}\n\n"
                    f"User Request: {user_query}\nOutput 5 concise interview questions."
                )
            else:
                frame.text = (
                    f"{SYSTEM_PROMPT}\n\n[DOCUMENT CONTEXT]:\n{context_text}\n\nUser Question: {user_query}"
                )

        await original_process_frame(frame, direction)

    # Override
    user_aggregator.process_frame = rag_process_frame

    # Run pipeline
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)

# -------------------
async def bot(runner_args: RunnerArguments):
    transport_params = {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)

# -------------------
if __name__ == "__main__":
    from pipecat.runner.run import main
    main()