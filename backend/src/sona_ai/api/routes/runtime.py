from fastapi import APIRouter

from sona_ai.core import runtime_devices


router = APIRouter()


@router.get("/runtime/devices")
def get_runtime_devices():
    return runtime_devices()
