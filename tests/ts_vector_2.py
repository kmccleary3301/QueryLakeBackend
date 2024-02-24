import time
from faker import Faker
from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine
from sqlalchemy import Column, DDL, event, text
from sqlalchemy.dialects.postgresql import TSVECTOR

class DocumentEmbeddingTSVectorTest1(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str = Field()
    ts_content: str = Field(sa_column=Column(TSVECTOR))  # Define ts_content as a str

engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")

# Create a trigger to update ts_content
trigger = DDL("""
CREATE OR REPLACE FUNCTION update_ts_content()
RETURNS TRIGGER AS $$
BEGIN
  NEW.ts_content := to_tsvector('english', NEW.text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_ts_content_trigger ON documentembeddingtsvectortest1;

CREATE TRIGGER update_ts_content_trigger
BEFORE INSERT OR UPDATE ON documentembeddingtsvectortest1
FOR EACH ROW EXECUTE FUNCTION update_ts_content();
""")
event.listen(DocumentEmbeddingTSVectorTest1.__table__, 'after_create', trigger.execute_if(dialect='postgresql'))

SQLModel.metadata.create_all(engine)

database = Session(engine)

def add_text(text_in : str):
    test_entry = DocumentEmbeddingTSVectorTest1(text=text_in)
    database.add(test_entry)
    database.commit()

def search_documents(session: Session, search_string: str, limit: int = 10):
    stmt = text("""
        SELECT id, text, ts_rank_cd(ts_content, websearch_to_tsquery(:search_string)) AS rank,
               ts_headline(text, websearch_to_tsquery(:search_string)) AS headline
        FROM documentembeddingtsvectortest1
        WHERE ts_content @@ websearch_to_tsquery(:search_string)
        ORDER BY rank DESC
        LIMIT :limit
    """).bindparams(search_string=search_string, limit=limit)

    results = session.exec(stmt)
    return results

# Stress test
fake = Faker()
num_entries = 10000  # Number of test entries to create
search_string = "test"  # Search string

# Create test entries
start_time = time.time()
for _ in range(num_entries):
    add_text(fake.text())
print(f"Inserted {num_entries} entries in {time.time() - start_time} seconds")

# Perform search query
start_time = time.time()
documents = search_documents(database, search_string, limit=10)
print(f"Performed search query in {time.time() - start_time} seconds")

for document in documents:
    print(document.id, document.text)

database.close()