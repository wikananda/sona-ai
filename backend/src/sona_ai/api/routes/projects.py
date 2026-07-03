import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from sona_ai.api.routes._project_helpers import _serialize_project, _serialize_recording
from sona_ai.api.schemas.projects import ProjectCreate
from sona_ai.core import setup_logging
from sona_ai.db.models import Project
from sona_ai.db.session import get_db
from sona_ai.storage import delete_project_dir


logger = setup_logging()
router = APIRouter()


@router.post("/projects")
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")

    project = Project(
        id=str(uuid.uuid4()),
        name=name,
        description=body.description.strip() if body.description else None,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _serialize_project(project)


@router.get("/projects")
def list_projects(db: Session = Depends(get_db)):
    projects = db.scalars(
        select(Project).order_by(Project.created_at.desc())
    ).all()
    return [_serialize_project(project) for project in projects]


@router.get("/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id)
        .options(selectinload(Project.recordings))
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    recordings = sorted(
        project.recordings,
        key=lambda recording: recording.created_at,
        reverse=True,
    )
    data = _serialize_project(project)
    data["recordings"] = [
        _serialize_recording(recording, include_transcript=False)
        for recording in recordings
    ]
    return data


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(project)
    db.commit()

    try:
        delete_project_dir(project_id)
    except Exception as exc:
        logger.warning("Failed to delete project audio directory: %s", exc)

    return {"ok": True}
