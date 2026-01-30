import os
from datetime import datetime
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
import uuid
from datetime import timedelta
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
# Tool Functions (REAL)
# =========================

async def check_availability(params: FunctionCallParams):
    table = dynamodb.Table(BOOKINGS_TABLE)

    date = params.arguments["date"]
    time = params.arguments["time"]

    # Build ISO start_time
    start_time_prefix = f"{date}T{time}"

    logger.info(f"[TOOL] check_availability scan: {start_time_prefix}")

    resp = table.scan(
        FilterExpression="begins_with(start_time, :st)",
        ExpressionAttributeValues={
            ":st": start_time_prefix
        }
    )

    is_available = resp.get("Count", 0) == 0

    await params.result_callback({
        "available": is_available
    })


async def create_booking(params: FunctionCallParams):
    table = dynamodb.Table(BOOKINGS_TABLE)

    date = params.arguments["date"]
    time = params.arguments["time"]
    name = params.arguments.get("name", "unknown")

    # Build ISO start time
    start_time_str = f"{date}T{time}:00"

    # Parse to datetime
    start_dt = datetime.fromisoformat(start_time_str)

    # Add 30 minutes
    end_dt = start_dt + timedelta(minutes=30)

    # Convert back to ISO strings
    start_time = start_dt.isoformat()
    end_time = end_dt.isoformat()

    appointment_id = str(uuid.uuid4())

    table.put_item(
        Item={
            "appointment_id": appointment_id,
            "start_time": start_time,
            "end_time": end_time,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
        }
    )

    await params.result_callback({
        "status": "confirmed",
        "appointment_id": appointment_id,
        "start_time": start_time,
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
    logger.info("Starting Booking Bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en",
    )

    llm = AWSBedrockLLMService(
        region_name=AWS_REGION,
        #model="anthropic.claude-3-sonnet-20240229-v1:0",
        model=os.getenv("BEDROCK_NOVA_PRO_PROFILE_ARN"),
        temperature=0.2,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
        tool_choice="auto",
    )

    # Register booking tools
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
            "date": {
                "type": "string",
                "description": "Appointment date in YYYY-MM-DD format",
            },
            "time": {
                "type": "string",
                "description": "Appointment time, e.g. 15:00",
            },
        },
        required=["date", "time"],
    )

    create_booking_function = FunctionSchema(
        name="create_booking",
        description="Create and save an appointment booking",
        properties={
            "date": {
                "type": "string",
                "description": "Appointment date in YYYY-MM-DD format",
            },
            "time": {
                "type": "string",
                "description": "Appointment time, e.g. 15:00",
            },
            "name": {
                "type": "string",
                "description": "Customer name",
            },
        },
        required=["date", "time"],
    )

    tools = ToolsSchema(
        standard_tools=[
            check_availability_function,
            create_booking_function,
        ]
    )

    # =========================
    # System Prompt (IMPORTANT)
    # =========================

    messages = [
        {
            "role": "system",
            "content": (
                "You are a voice appointment booking assistant. "
                "When a user wants to book an appointment, you MUST "
                "first call check_availability with the requested date and time. "
                "If available, confirm and then call create_booking. "
                "If not available, politely ask for another date or time. "
                "Speak naturally and briefly."
            ),
        },
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
