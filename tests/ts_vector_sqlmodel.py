# from typing import Optional, List, Literal
# from sqlmodel import Field, SQLModel, ARRAY, String, Integer, Float, JSON, LargeBinary

# from sqlalchemy.sql.schema import Column
# from sqlalchemy.dialects.postgresql import TSVECTOR
# from sqlalchemy import event, text

# from sqlalchemy.sql import func

# from sqlmodel import Session, create_engine
# from pgvector.sqlalchemy import Vector
# from sqlalchemy import Column

# class DocumentEmbeddingTSVectorTest1(SQLModel, table=True):
#     id: Optional[int] = Field(default=None, primary_key=True)
#     text: str = Field()
#     ts_content: TSVECTOR = Field(sa_column=Column(TSVECTOR))
    
# engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
        
# SQLModel.metadata.create_all(engine)
# database = Session(engine)

# # Create a test entry with ts_content
# test_entry = DocumentEmbeddingTSVectorTest1(text="Test entry", ts_content="example ts_content")
# database.add(test_entry)
# database.commit()

# print(test_entry)


print("Test 1")

from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine
from sqlalchemy import Column, DDL, event
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
print("Test 1.2")
SQLModel.metadata.create_all(engine)

database = Session(engine)
print("Test 1.3")
# Create a test entry



def add_text(text_in : str):
    test_entry = DocumentEmbeddingTSVectorTest1(text=text_in)
    database.add(test_entry)
    database.commit()
    print(test_entry.text, test_entry.ts_content)

add_text("The powerhouse of the cell is the mitochondria")
add_text("The pinnacle of the mind is the membrane")
add_text("The point of the body is the GPU core")
add_text("The week of the farm is the cytoplasm")

print("Test 1.6")
# database.close()

print("Test 1.7")




from sqlalchemy import text
from sqlmodel import Session, create_engine
from typing import List

# def search_documents(session: Session, search_string: str) -> List[DocumentEmbeddingTSVectorTest1]:
#     # Create a SELECT statement that includes a ts_query clause
#     stmt = text("""
#         SELECT * FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ to_tsquery(:search_string)
#     """).bindparams(search_string=search_string)

#     # Execute the statement and fetch all results
#     results = session.execute(stmt).scalars().all()

#     return results

# # Usage
# engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
# session = Session(engine)

# search_string = "your search string"
# documents = search_documents(session, search_string)

# for document in documents:
#     print(document.id, document.text)

# session.close()

print("Test 2")

# def search_documents(session: Session, search_string: str) -> List[DocumentEmbeddingTSVectorTest1]:
#     # Split the search string into words and combine them with the & operator
#     tsquery_string = " | ".join(search_string.split())

#     # Create a SELECT statement that includes a ts_query clause
#     stmt = text("""
#         SELECT * FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ plainto_tsquery(:tsquery_string)
#     """).bindparams(tsquery_string=tsquery_string)

#     # Execute the statement and fetch all results
#     # results = session.execute(stmt).scalars().all()
    
#     results = session.exec(stmt).all()

#     print(results)
    
#     return results




# def search_documents(session: Session, search_string: str, limit: int = 10) -> List[DocumentEmbeddingTSVectorTest1]:
#     # Split the search string into words and combine them with the | operator
#     tsquery_string = " | ".join(search_string.split())

#     # Create a SELECT statement that includes a ts_query clause and orders the results by ts_rank_cd
#     stmt = text("""
#         SELECT *, ts_rank_cd(ts_content, plainto_tsquery(:tsquery_string)) AS rank
#         FROM documentembeddingtsvectortest1
#         WHERE ts_content @@ websearch_to_tsquery('english', :tsquery_string)
#         ORDER BY rank DESC
#         LIMIT :limit
#     """).bindparams(tsquery_string=tsquery_string, limit=limit)

#     # Execute the statement and fetch all results
#     results = session.exec(stmt)

#     return results


def search_documents(session: Session, search_string: str, limit: int = 10):
    # Create a SELECT statement that includes a ts_query clause and orders the results by ts_rank_cd
    stmt = text("""
        SELECT id, text, ts_rank_cd(ts_content, websearch_to_tsquery(:search_string)) AS rank,
               ts_headline(text, websearch_to_tsquery(:search_string)) AS headline
        FROM documentembeddingtsvectortest1
        WHERE ts_content @@ websearch_to_tsquery(:search_string)
        ORDER BY rank DESC
        LIMIT :limit
    """).bindparams(search_string=search_string, limit=limit)

    # Execute the statement and fetch all results
    
    print(session.exec(stmt))
    
    results = session.exec(stmt)

    return results



print("Test 2.2")
# Usage
# engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
# session = Session(engine)


print("Test 2.3")
search_string = "Is the powerhouse of the cell the GPU core?"
documents = search_documents(database, search_string, limit=10)
print(documents)
print("Test 2.4\n\n\n\n")

for e in documents.scalars():
    print("e:", e)

for document in documents:
    print(document.id, document.text)
print("\n\nTest 2.5")

database.close()