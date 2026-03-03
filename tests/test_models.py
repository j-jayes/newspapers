"""Tests for the Pydantic data models."""

import pytest
from pydantic import ValidationError

from newspapers.models import JobAdvertisement, NewspaperPageMeta, PageSegment


class TestJobAdvertisement:
    """Validate the JobAdvertisement schema."""

    def test_minimal_valid(self):
        ad = JobAdvertisement(
            job_title="Springpojke",
            confidence_score=0.95,
        )
        assert ad.job_title == "Springpojke"
        assert ad.skills_required == []
        assert ad.gender_preference is None
        assert ad.confidence_score == 0.95

    def test_full_valid(self):
        ad = JobAdvertisement(
            job_title="Hushållerska",
            skills_required=["matlagning", "städning"],
            gender_preference="kvinna",
            age_requirement="20-35 år",
            location="Stockholm",
            employer="Familjen Andersson",
            compensation="Fri kost och logi",
            confidence_score=0.87,
        )
        assert len(ad.skills_required) == 2
        assert ad.employer == "Familjen Andersson"

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            JobAdvertisement(job_title="Test", confidence_score=1.5)

    def test_confidence_negative(self):
        with pytest.raises(ValidationError):
            JobAdvertisement(job_title="Test", confidence_score=-0.1)

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            JobAdvertisement()


class TestPageSegment:
    """Validate the PageSegment schema."""

    def test_valid_segment(self):
        seg = PageSegment(
            label="job_advertisement",
            x_min=10.0,
            y_min=20.0,
            x_max=200.0,
            y_max=300.0,
            confidence=0.92,
        )
        assert seg.label == "job_advertisement"
        assert seg.x_max > seg.x_min

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            PageSegment(
                label="headline",
                x_min=0,
                y_min=0,
                x_max=100,
                y_max=100,
                confidence=1.1,
            )


class TestNewspaperPageMeta:
    """Validate the NewspaperPageMeta schema."""

    def test_valid_meta(self):
        meta = NewspaperPageMeta(
            source_id="bib13991099_19030212",
            newspaper_title="Dagens Nyheter",
            publication_date="1903-02-12",
            page_number=4,
            image_path="/data/raw/bib13991099_19030212.png",
        )
        assert meta.newspaper_title == "Dagens Nyheter"
        assert meta.page_number == 4

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            NewspaperPageMeta(
                source_id="test",
                newspaper_title="DN",
            )
