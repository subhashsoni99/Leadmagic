import boto3
from datetime import datetime, timedelta
import uuid

dynamodb = boto3.resource("dynamodb", region_name="us-west-2")
table = dynamodb.Table("appointments")

item = {
    "appointment_id": str(uuid.uuid4()),
    "start_time": datetime.now().isoformat(),
    "end_time": (datetime.now() + timedelta(hours=1)).isoformat(),
    "created_at": datetime.utcnow().isoformat(),
}

resp = table.put_item(Item=item)
print(resp)
