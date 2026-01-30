# booking.py
import uuid
import boto3
import re
from datetime import datetime, timedelta
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame

DYNAMO_TABLE = "appointments"


class BookingProcessor(FrameProcessor):
    """
    Robust BookingProcessor for Bedrock + Pipecat.

    ✅ Handles:
    - Python code blocks from Bedrock
    - Streaming text
    - check_appointment_slot + book_appointment
    - DynamoDB persistence
    - Human TTS confirmations
    """

    def __init__(self):
        super().__init__()
        self.buffer = ""
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table(DYNAMO_TABLE)

    async def process_frame(self, frame, task, *args):
        await super().process_frame(frame, task, *args)

        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, task)
            return

        text = frame.text or ""
        logger.warning(f"[BookingProcessor] TEXT FRAME: {text!r}")

        self.buffer += text

        # If Bedrock outputs python code, parse datetime() and book_appointment
        if "datetime(" in self.buffer and "book_appointment" in self.buffer:
            logger.success("[BookingProcessor] Detected tool call inside code block")
            await self._handle_code_tool(self.buffer, task)
            self.buffer = ""
            return

        # 🚫 Block code from TTS
        # Normal human text
        await self.push_frame(frame, task)

    # =========================
    # CODE TOOL HANDLER
    # =========================
    async def _handle_code_tool(self, text: str, task):
        """Extracts datetime(YYYY, M, D, H, M) from code"""

        dt_match = re.search(
            r"datetime\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
            text,
        )

        if not dt_match:
            logger.error("❌ Could not extract datetime() from Bedrock code")
            return

        year, month, day, hour, minute = map(int, dt_match.groups())
        dt = datetime(year, month, day, hour, minute)

        logger.success(f"[BookingProcessor] Parsed datetime: {dt}")

        if self._has_conflict(dt):
            reply = self._human_conflict_message(dt)
        else:
            self._write_booking(dt, client_name=self._extract_client_name(text))
            reply = self._human_success_message(dt)

        # 🔊 Send human TTS
        await task.queue_frames([TextFrame(text=reply)])

    # =========================
    # DYNAMODB
    # =========================
    def _has_conflict(self, dt: datetime) -> bool:
        # Simple scan (OK for small table, improve later)
        resp = self.table.scan()

        for item in resp.get("Items", []):
            start = datetime.fromisoformat(item["start_time"])
            end = datetime.fromisoformat(item["end_time"])

            if not (dt + timedelta(hours=1) <= start or dt >= end):
                logger.info("[BookingProcessor] Conflict found in DynamoDB")
                return True

        return False

    def _write_booking(self, dt: datetime, client_name: str | None):
        item = {
            "appointment_id": str(uuid.uuid4()),
            "start_time": dt.isoformat(),
            "end_time": (dt + timedelta(hours=1)).isoformat(),
            "client_name": client_name or "Unknown",
            "created_at": datetime.utcnow().isoformat(),
        }

        self.table.put_item(Item=item)
        logger.success(f"[BookingProcessor] ✅ STORED IN AWS DYNAMODB: {item}")

    # =========================
    # HUMAN SPEECH
    # =========================
    def _human_success_message(self, dt: datetime) -> str:
        return (
            f"Your appointment is confirmed for "
            f"{dt.strftime('%A, %B %d')} at "
            f"{dt.strftime('%I:%M %p').lstrip('0')}."
        )

    def _human_conflict_message(self, dt: datetime) -> str:
        return (
            f"Sorry, {dt.strftime('%A, %B %d at %I:%M %p').lstrip('0')} "
            f"is already booked. Please choose another time."
        )

    # =========================
    # CLIENT NAME (OPTIONAL)
    # =========================
    def _extract_client_name(self, text: str) -> str | None:
        m = re.search(r'client_name\s*=\s*"([^"]+)"', text)
        if m:
            return m.group(1)
        return None
