"""Pydantic models for structured data extraction from historical job ads."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class JobAdvertisement(BaseModel):
    """Schema for a single historical Swedish job advertisement.

    Extracted via Vision LLM from a cropped newspaper image segment,
    bypassing traditional OCR to handle mixed Fraktur/Antiqua scripts.
    """

    job_title: str = Field(
        ...,
        description="The advertised job title, in the original Swedish.",
    )
    skills_required: list[str] = Field(
        default_factory=list,
        description="Skills or qualifications mentioned in the advertisement.",
    )
    gender_preference: Optional[str] = Field(
        default=None,
        description="Gender preference stated in the ad, if any.",
    )
    age_requirement: Optional[str] = Field(
        default=None,
        description="Age requirement or preference stated in the ad, if any.",
    )
    location: Optional[str] = Field(
        default=None,
        description="Location associated with the job, if mentioned.",
    )
    employer: Optional[str] = Field(
        default=None,
        description="Employer or firm name, if mentioned.",
    )
    compensation: Optional[str] = Field(
        default=None,
        description="Compensation details (salary, room & board, etc.), if mentioned.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the extraction (0.0–1.0).",
    )


class PageSegment(BaseModel):
    """A detected region on a newspaper page produced by the segmentation model."""

    label: str = Field(
        ...,
        description="Class label (e.g. 'job_advertisement', 'article_text', 'headline').",
    )
    x_min: float = Field(..., description="Left edge of the bounding box (pixels).")
    y_min: float = Field(..., description="Top edge of the bounding box (pixels).")
    x_max: float = Field(..., description="Right edge of the bounding box (pixels).")
    y_max: float = Field(..., description="Bottom edge of the bounding box (pixels).")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score.",
    )


class NewspaperPageMeta(BaseModel):
    """Metadata for a single scanned newspaper page from the KB archive."""

    source_id: str = Field(
        ...,
        description="Unique identifier from the KB archive.",
    )
    newspaper_title: str = Field(
        ...,
        description="Title of the newspaper (e.g. 'Dagens Nyheter').",
    )
    publication_date: str = Field(
        ...,
        description="Publication date in ISO-8601 format (YYYY-MM-DD).",
    )
    page_number: int = Field(..., description="Page number within the issue.")
    image_path: str = Field(
        ...,
        description="Path to the preprocessed image file on disk.",
    )
