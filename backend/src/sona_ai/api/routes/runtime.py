from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from sona_ai.core import runtime_devices
from sona_ai.services.model_download_service import model_download_service


router = APIRouter()


@router.get("/runtime/devices")
def get_runtime_devices():
    return runtime_devices()


@router.get("/runtime/models")
def list_runtime_models():
    return model_download_service.list_models()


@router.post("/runtime/models/{model_id}/download")
def download_runtime_model(model_id: str):
    try:
        return asdict(model_download_service.start_download(model_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc


@router.get("/runtime/model-downloads/{job_id}")
def get_model_download_job(job_id: str):
    try:
        return asdict(model_download_service.get_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model download job: {job_id}") from exc
