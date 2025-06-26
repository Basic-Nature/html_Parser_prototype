from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    DateTime, 
    Text, 
    JSON, 
    ForeignKey, 
    Boolean, 
    Float, 
    LargeBinary, 
    create_engine
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime, timezone

Base = declarative_base()

class Contest(Base):
    __tablename__ = 'contests'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    year = Column(Integer)
    type = Column(String)
    state = Column(String)
    county = Column(String)
    # metadata = Column(JSON)

class TableStructure(Base):
    __tablename__ = 'table_structures'
    id = Column(Integer, primary_key=True)
    contest_title = Column(String, nullable=False, index=True)
    headers = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    confirmed_by_user = Column(Boolean, default=False)
    ml_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

class BatchMetadata(Base):
    __tablename__ = 'batch_metadata'
    batch_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String)
    started_at = Column(DateTime, default=datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    status = Column(String)

class StagingElectionResult(Base):
    __tablename__ = 'staging_election_results'
    id = Column(Integer, primary_key=True)
    batch_id = Column(UUID(as_uuid=True), ForeignKey('batch_metadata.batch_id'), nullable=False)
    state = Column(String)
    county = Column(String)
    source_url = Column(String)
    raw_html = Column(Text)
    parsed_at = Column(DateTime, default=datetime.now(timezone.utc))
    status = Column(String, default='pending')

class WarehouseElectionResult(Base):
    __tablename__ = 'warehouse_election_results'
    id = Column(Integer, primary_key=True)
    batch_id = Column(UUID(as_uuid=True), ForeignKey('batch_metadata.batch_id'), nullable=False)
    state = Column(String)
    county = Column(String)
    contest_title = Column(String)
    candidate = Column(String)
    party = Column(String)
    votes = Column(Integer)
    precinct = Column(String)
    election_date = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.now(timezone.utc))

class Entity(Base):
    __tablename__ = 'entities'
    id = Column(Integer, primary_key=True)
    entity_type = Column(String, nullable=False)
    value = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

class EmbeddingCache(Base):
    __tablename__ = 'embeddings'
    segment_hash = Column(String, primary_key=True)
    embedding = Column(LargeBinary)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    context = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

# Add more models as needed for your project (e.g., users, logs, etc.)
if __name__ == "__main__":
    # Import POSTGRES_URL from config
    from ..config import POSTGRES_URL
    engine = create_engine(POSTGRES_URL)
    print(f"Creating all tables in: {POSTGRES_URL}")
    Base.metadata.create_all(engine)
    print("All tables created.")