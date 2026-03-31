"""Tests for database models in utils/models.py"""
import pytest
from sqlalchemy.exc import IntegrityError
from webapp.parser.utils.models import (
    Contest,
    Candidate,
    Party,
    State,
    County,
    Result,
)


class TestModels:
    """Tests for SQLAlchemy models."""
    
    def test_contest_creation(self, db_session):
        """Test Contest model creation."""
        contest = Contest(
            title="U.S. Representative District 3",
            year=2024,
            type_="General"
        )
        db_session.add(contest)
        db_session.commit()
        
        assert contest.id is not None
        assert contest.title == "U.S. Representative District 3"
    
    def test_party_creation(self, db_session):
        """Test Party model creation."""
        party = Party(name="Democratic", abbreviation="DEM")
        db_session.add(party)
        db_session.commit()
        
        assert party.id is not None
        assert party.abbreviation == "DEM"
    
    def test_state_county_relationship(self, db_session):
        """Test State-County relationship."""
        state = State(name="New York")
        county = County(name="Rockland", state=state)
        
        db_session.add(state)
        db_session.add(county)
        db_session.commit()
        
        assert county.state.name == "New York"
        assert state.id is not None

    def test_county_unique_per_state_constraint(self, db_session):
        state = State(name="California")
        db_session.add(state)
        db_session.flush()

        db_session.add(County(name="Alameda", state=state))
        db_session.flush()
        db_session.add(County(name="Alameda", state=state))

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

    def test_contest_unique_constraint_same_state_county_year(self, db_session):
        state = State(name="Texas")
        county = County(name="Harris", state=state)
        db_session.add_all([state, county])
        db_session.flush()

        db_session.add(Contest(title="Governor", year=2024, type_="General", state=state, county=county))
        db_session.flush()
        db_session.add(Contest(title="Governor", year=2024, type_="General", state=state, county=county))

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

    def test_candidate_result_relationship(self, db_session):
        party = Party(name="Democratic", abbreviation="DEM")
        candidate = Candidate(name="Jane Doe", party=party)
        contest = Contest(title="Mayor", year=2024, type_="General")
        result = Result(candidate=candidate, contest=contest, votes=1234, percent=51.2)

        db_session.add_all([party, candidate, contest, result])
        db_session.commit()

        assert candidate.results[0].votes == 1234
        assert contest.results[0].candidate.name == "Jane Doe"
