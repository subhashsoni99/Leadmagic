import os
import uuid
from datetime import datetime, timedelta, date as date_cls

import boto3
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.deepgram.tts import DeepgramTTSService

load_dotenv(override=True)

# =========================
# AWS DynamoDB Setup
# =========================

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
BOOKINGS_TABLE = os.getenv("BOOKINGS_TABLE", "appointments")

dynamodb = boto3.resource(
    "dynamodb",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
)

# =========================
# Helpers
# =========================

def normalize_date_time(date_str: str, time_str: str) -> tuple[str, str]:
    """
    Guarantees current year and proper ISO formatting.
    Returns (YYYY-MM-DD, HH:MM)
    """
    today = date_cls.today()

    parts = date_str.split("-")
    if len(parts) == 3:
        year, month, day = parts
        if int(year) < today.year:
            year = str(today.year)
    else:
        year = str(today.year)
        month = f"{today.month:02d}"
        day = f"{today.day:02d}"

    clean_date = f"{year}-{month}-{day}"
    clean_time = time_str[:5]

    return clean_date, clean_time


def humanize_datetime(date_str: str, time_str: str) -> str:
    """
    Convert ISO to natural spoken English for TTS.
    Example: 2026-01-30 + 10:00 -> January 30th at 10 AM
    """
    dt = datetime.fromisoformat(f"{date_str}T{time_str}:00")

    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    spoken_date = dt.strftime(f"%B {day}{suffix}")
    spoken_time = dt.strftime("%-I:%M %p").replace(":00", "")

    return f"{spoken_date} at {spoken_time}"


# =========================
# Tool Functions
# =========================

async def check_availability(params: FunctionCallParams):
    table = dynamodb.Table(BOOKINGS_TABLE)

    raw_date = params.arguments["date"]
    raw_time = params.arguments["time"]

    date_key, time_key = normalize_date_time(raw_date, raw_time)

    logger.info(f"[TOOL] check_availability: {date_key} {time_key}")

    resp = table.get_item(
        Key={
            "date": date_key,
            "time": time_key,
        }
    )

    is_available = "Item" not in resp

    await params.result_callback({
        "available": is_available
    })


async def create_booking(params: FunctionCallParams):
    table = dynamodb.Table(BOOKINGS_TABLE)

    raw_date = params.arguments["date"]
    raw_time = params.arguments["time"]
    name = params.arguments.get("name", "unknown")

    date_key, time_key = normalize_date_time(raw_date, raw_time)

    start_dt = datetime.fromisoformat(f"{date_key}T{time_key}:00")
    end_dt = start_dt + timedelta(minutes=30)

    appointment_id = str(uuid.uuid4())
    spoken_when = humanize_datetime(date_key, time_key)

    logger.info(f"[TOOL] create_booking: {date_key} {time_key} {name}")

    table.put_item(
        Item={
            "date": date_key,
            "time": time_key,
            "appointment_id": appointment_id,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
        },
        ConditionExpression="attribute_not_exists(#d) AND attribute_not_exists(#t)",
        ExpressionAttributeNames={
            "#d": "date",
            "#t": "time",
        }
    )

    await params.result_callback({
        "status": "confirmed",
        "appointment_id": appointment_id,
        "date": date_key,
        "time": time_key,
        "start_time": start_dt.isoformat(),
        "spoken_when": spoken_when,   # 👈 HUMAN SPEECH
    })


# =========================
# Transport Params
# =========================

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}

# =========================
# Main Bot Logic
# =========================

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Booking Bot (Human Voice Mode)")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en",
    )

    llm = AWSBedrockLLMService(
        region_name=AWS_REGION,
        model=os.getenv("BEDROCK_NOVA_PRO_PROFILE_ARN"),
        temperature=0.2,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
        tool_choice="auto",
    )

    llm.register_function("check_availability", check_availability)
    llm.register_function("create_booking", create_booking)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check that for you."))

    # =========================
    # Tool Schemas
    # =========================

    check_availability_function = FunctionSchema(
        name="check_availability",
        description="Check if an appointment slot is available",
        properties={
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM"},
        },
        required=["date", "time"],
    )

    create_booking_function = FunctionSchema(
        name="create_booking",
        description="Create and save an appointment booking",
        properties={
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM"},
            "name": {"type": "string", "description": "Customer name"},
        },
        required=["date", "time"],
    )

    tools = ToolsSchema(
        standard_tools=[check_availability_function, create_booking_function]
    )

    # =========================
    # System Prompt (VOICE SAFE)
    # =========================

    today_str = date_cls.today().isoformat()

    messages = [
        {
            "role": "system",
            "content": (
                f"Today is {today_str}. "
                "You are a voice appointment booking assistant. "

                "IMPORTANT: "
                "Use ISO format ONLY for tool calls. "
                "For speaking to the user, ALWAYS use natural human language. "
                "Never read dates like 2026-01-30 or times like 10:00. "
                "Always say them like 'January 30th at 10 AM'. "

                "If the user gives a date without a year, assume the current year. "
                "Resolve relative dates like tomorrow based on today. "

                "When booking, you MUST first call check_availability. "
                "If available, then call create_booking. "
                "If not available, politely ask for another time. "
                "Speak like a friendly human receptionist."
            ),
        },
        {"role": "user", "content": "Hello"}
    ]

    context = LLMContext(messages, tools)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
