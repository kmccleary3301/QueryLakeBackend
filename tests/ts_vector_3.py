import time
import random
from faker import Faker
from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine
from sqlalchemy import Column, DDL, event, text
from sqlalchemy.dialects.postgresql import TSVECTOR
from typing import List, Tuple

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

CREATE INDEX IF NOT EXISTS ts_content_gin ON documentembeddingtsvectortest1 USING gin(ts_content);
""")
event.listen(DocumentEmbeddingTSVectorTest1.__table__, 'after_create', trigger.execute_if(dialect='postgresql'))

SQLModel.metadata.create_all(engine)

database = Session(engine)

def add_text(text_in : str):
    test_entry = DocumentEmbeddingTSVectorTest1(text=text_in)
    database.add(test_entry)
    database.commit()


# # Version 1
# def search_documents(session: Session, search_string: str, limit: int = 10):
#     stmt = text("""
#         SELECT id, text, ts_rank_cd(ts_content, websearch_to_tsquery(:search_string)) AS rank,
#                ts_headline(text, websearch_to_tsquery(:search_string)) AS headline
#         FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ websearch_to_tsquery(:search_string)
#         ORDER BY rank DESC
#         LIMIT :limit
#     """).bindparams(search_string=search_string, limit=limit)

#     results = session.exec(stmt)
    
#     print(results)
#     return results


# # Version 2
# def search_documents(session: Session, 
#                      search_string: str, 
#                      limit: int = 10) -> List[Tuple[
#                          int,  # id
#                          str,  # text
#                          float,  # rank
#                          str  # headline
#                      ]]:
#     stmt = text("""
#         SELECT *, ts_rank_cd(ts_content, websearch_to_tsquery(:search_string)) AS rank,
#                ts_headline(text, websearch_to_tsquery(:search_string)) AS headline
#         FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ websearch_to_tsquery(:search_string)
#         ORDER BY rank DESC
#         LIMIT :limit
#     """).bindparams(search_string=search_string, limit=limit)

#     results = session.exec(stmt)
    
    
#     print("\nRESULTS")
#     print(results)
#     return results

# # Version 3
# def search_documents(session: Session, search_string: str, limit: int = 10):
#     # Split the search string into substrings
#     substrings = search_string.split(' ')
    
#     # Create a tsquery for each substring
#     tsqueries = [f"websearch_to_tsquery('{substring}')" for substring in substrings]
    
#     # Combine the tsqueries with the OR operator
#     tsquery = ' || '.join(tsqueries)
    
#     stmt = text(f"""
#         SELECT id, text, ts_rank_cd(ts_content, {tsquery}) AS rank,
#                ts_headline(text, {tsquery}) AS headline
#         FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ {tsquery}
#         ORDER BY rank DESC
#         LIMIT :limit
#     """).bindparams(limit=limit)

#     results = session.exec(stmt)
#     return results

# # Version 4
# def search_documents(session: Session, search_string: str, limit: int = 10):
#     # Split the search string into substrings
#     substrings = search_string.split(' ')
    
#     # Create a tsquery for each substring
#     tsqueries = [f"websearch_to_tsquery('{substring}')" for substring in substrings]
    
#     # Combine the tsqueries with the OR operator
#     tsquery = ' | '.join(tsqueries)
    
#     stmt = text(f"""
#         SELECT id, text, ts_rank_cd(ts_content, {tsquery}) AS rank,
#                ts_headline(text, {tsquery}) AS headline
#         FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ {tsquery}
#         ORDER BY ts_rank_cd(ts_content, {tsquery}) DESC
#         LIMIT :limit
#     """).bindparams(limit=limit)

#     results = session.exec(stmt)
#     return results

# # Version 5
# def search_documents(session: Session, search_string: str, limit: int = 10):
#     # Split the search string into substrings
#     substrings = search_string.split(' ')
    
#     # Create a tsquery for each substring
#     tsqueries = [f"websearch_to_tsquery('{substring}')" for substring in substrings]
    
#     # Combine the tsqueries with the OR operator
#     tsquery = ' | '.join(tsqueries)
    
#     stmt = text(f"""
#         SELECT id, text, ts_rank_cd(ts_content, {tsquery}) AS rank,
#                ts_headline(text, {tsquery}) AS headline
#         FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ {tsquery}
#         ORDER BY ts_rank_cd(ts_content, {tsquery}) DESC
#         LIMIT :limit
#     """).bindparams(limit=limit)

#     results = session.exec(stmt)
#     return results

# # Version 5
# def search_documents(session: Session, search_string: str, limit: int = 10):
#     # Split the search string into substrings
#     substrings = search_string.split(' ')
    
#     # Create a search string for each substring
#     search_strings = [f"'{substring}'" for substring in substrings]
    
#     # Combine the search strings with the OR operator
#     search_string = ' || '.join(search_strings)
    
#     stmt = text(f"""
#         SELECT id, text, ts_rank_cd(ts_content, websearch_to_tsquery({search_string})) AS rank,
#                ts_headline(text, websearch_to_tsquery({search_string})) AS headline
#         FROM documentembeddingtsvectortest1
#         ORDER BY ts_rank_cd(ts_content, websearch_to_tsquery({search_string})) DESC
#         LIMIT :limit
#     """).bindparams(limit=limit)

#     results = session.exec(stmt)
#     return results

# Version 6
def search_documents(session: Session, search_string: str, limit: int = 10):
    # Replace spaces with the OR operator
    search_string = search_string.replace(' ', ' | ')
    
    stmt = text("""
        SELECT id, text, ts_rank_cd(ts_content, query) AS rank,
               ts_headline(text, query) AS headline
        FROM documentembeddingtsvectortest1,
             to_tsquery(:search_string) query
        WHERE ts_content @@ query
        ORDER BY rank DESC
        LIMIT :limit
    """).bindparams(search_string=search_string, limit=limit)

    results = session.exec(stmt)
    return results

# Stress test
fake = Faker()
num_entries = 10000  # Number of test entries to create
common_phrase = "mario kart 64 rainbow road"  # Common phrase to insert into a subset of entries
other_phrase = "luigi kart 64 toad's turnpike"  # Common phrase to insert into a subset of entries

search_string = "peach kart 64 rainbow turnpike"  # Search string

# Create test entries
start_time = time.time()
for _ in range(num_entries):
    sentence = fake.sentence()
    insert_position = random.randint(0, len(sentence))
    if random.random() < 0.005:  # Insert common phrase into a subset of entries (0.5%)
        text_make = sentence[:insert_position] + f" {common_phrase} " + sentence[insert_position:]
    elif random.random() < 0.01:  # Insert other phrase into a subset of entries (0.5%)
        text_make = sentence[:insert_position] + f" {other_phrase} " + sentence[insert_position:]
    else:
        text_make = sentence
    add_text(text_make)
print(f"Inserted {num_entries} entries in {time.time() - start_time} seconds")

# Perform search query
start_time = time.time()
documents = search_documents(database, search_string, limit=10)
print(f"Performed search query in {time.time() - start_time} seconds")

for document in documents:
    print("\nDOCUMENT ENTRY")
    print(document)
    print(document.id, document.text)

database.close()