#!/usr/bin/env python3
"""
Database Initialization for SMART Elections Workflow

Initializes SQLAlchemy models including:
- DownloadRecord (Worklist)
- ValidationRecord_DL1 (Human-curated)
- ValidationRecord_DL2 (Machine-enriched)
- PreQCComparison (Strict + Fuzzy comparison results)
- QC1Checkpoint (QC1 designee review)
- QC2Checkpoint (QC2 final review)
- ChainOfCustody (Complete audit trail)

Usage:
    python db_init.py                    # Use DATABASE_URL env var
    python db_init.py sqlite:///test.db  # Use local SQLite
    python db_init.py postgresql://...   # Use PostgreSQL connection string
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from models.election_data import Base

def get_connection_string():
    """Get database connection string from argument or env var."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    db_url = os.getenv('DATABASE_URL', 'sqlite:///election_data.db')
    
    # Allow relative paths for SQLite
    if db_url.startswith('sqlite:///'):
        # Convert to absolute path
        rel_path = db_url.replace('sqlite:///', '')
        abs_path = os.path.abspath(rel_path)
        db_url = f'sqlite:///{abs_path}'
    
    return db_url

def init_db(db_url=None):
    """Initialize database and create all tables."""
    if db_url is None:
        db_url = get_connection_string()
    
    print(f"Connecting to database: {db_url}")
    
    try:
        # Create engine
        engine = create_engine(db_url, echo=False)
        
        # Verify connection
        with engine.connect() as conn:
            print("✓ Database connection successful")
        
        # Inspect existing tables
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        print(f"\nExisting tables: {len(existing_tables)}")
        if existing_tables:
            for table in sorted(existing_tables):
                print(f"  - {table}")
        
        # Create all tables
        print("\nCreating tables...")
        Base.metadata.create_all(engine)
        
        # Verify new tables
        inspector = inspect(engine)
        new_tables = inspector.get_table_names()
        print(f"✓ Tables after initialization: {len(new_tables)}")
        if new_tables:
            for table in sorted(new_tables):
                print(f"  - {table}")
        
        # Verify indexes were created
        print("\nVerifying indexes...")
        for table_name in ["download_record", "validation_record_dl1", "validation_record_dl2"]:
            if table_name in new_tables:
                indexes = inspector.get_indexes(table_name)
                if indexes:
                    print(f"  {table_name}: {len(indexes)} indexes")
                    for idx in indexes:
                        print(f"    - {idx['name']}: {', '.join(idx['column_names'])}")
        
        print("\n✓ Database initialization complete!")
        
        # Show schema summary
        print("\n" + "="*70)
        print("SCHEMA SUMMARY - SMART Elections Workflow Models")
        print("="*70)
        
        schema_info = {
            "download_record": "Worklist tracking all 4 steps per race",
            "validation_record_dl1": "Human-curated ground truth data",
            "validation_record_dl2": "Machine-enriched data from Google Sheets",
            "preqc_comparison": "Strict equality + fuzzy match results",
            "qc1_checkpoint": "QC1 designee review and approval",
            "qc2_checkpoint": "QC2 final review and export approval",
            "chain_of_custody": "Complete audit trail of all changes",
        }
        
        for table_name, description in schema_info.items():
            if table_name in new_tables:
                print(f"✓ {table_name:30} - {description}")
        
        print("="*70)
        
        return True
    
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_connection(db_url=None):
    """Test database connection and basic operations."""
    if db_url is None:
        db_url = get_connection_string()
    
    print(f"\nTesting database operations...")
    
    try:
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Try to query a table
        from models.election_data import DownloadRecord
        count = session.query(DownloadRecord).count()
        print(f"✓ DownloadRecord query successful (count: {count})")
        
        session.close()
        return True
    
    except Exception as e:
        print(f"✗ Error testing database: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SMART Elections Database Initialization")
    print("="*70 + "\n")
    
    success = init_db()
    
    if success:
        test_connection()
        print("\n✓ Ready to use!\n")
        sys.exit(0)
    else:
        print("\n✗ Initialization failed\n")
        sys.exit(1)
