import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import json
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, text
import openai
from typing import Dict, List
import os
import sqlite3
from pathlib import Path

# MongoDB connection
def connect_to_mongodb():
    client = MongoClient("MongoDBCluster")
    db = client["sql_schemas"]
    return db

def infer_sql_type(dtype):
    if "int" in str(dtype):
        return Integer
    elif "float" in str(dtype):
        return Float
    elif "datetime" in str(dtype):
        return DateTime
    else:
        return String

def generate_sql_schema(df: pd.DataFrame, table_name: str) -> Dict:
    """Generate SQL schema from DataFrame"""
    columns = []
    for col_name, dtype in df.dtypes.items():
        sql_type = infer_sql_type(dtype)
        # Convert numpy.bool_ to Python bool
        is_nullable = bool(df[col_name].isnull().any())
        columns.append({
            "name": col_name,
            "type": str(sql_type.__name__),
            "nullable": is_nullable
        })
    
    return {
        "table_name": table_name,
        "columns": columns,
        "created_at": datetime.now().isoformat()  # Convert datetime to string
    }

def convert_to_mongodb_compatible(obj):
    """Convert numpy/pandas types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_to_mongodb_compatible(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_mongodb_compatible(item) for item in obj]
    elif isinstance(obj, np.bool_):  # Updated to only use np.bool_
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def store_schema_in_mongodb(schema: Dict, db):
    """Store schema in MongoDB"""
    collection = db["schemas"]
    # Convert all values to MongoDB-compatible types
    compatible_schema = convert_to_mongodb_compatible(schema)
    collection.insert_one(compatible_schema)

def setup_sqlite_db():
    """Create SQLite database connection"""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create database file in the data directory
    db_path = data_dir / "sql_data.db"
    conn = sqlite3.connect(db_path)
    
    # Enable foreign key support
    conn.execute("PRAGMA foreign_keys = ON")
    
    return conn


# Schema Agent class
class SchemaSelectionAgent:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
    
    def format_schema_info(self, schemas: List[Dict]) -> str:
        """Format schema information for the agent"""
        schema_info = "Available tables and their columns:\n\n"
        for schema in schemas:
            schema_info += f"Table: {schema['table_name']}\n"
            schema_info += "Columns:\n"
            for column in schema['columns']:
                schema_info += f"- {column['name']} ({column['type']})\n"
            schema_info += "\n"
        return schema_info

    def select_schemas(self, user_question: str, available_schemas: List[Dict]) -> List[str]:
        """Select relevant schemas based on user question"""
        schema_info = self.format_schema_info(available_schemas)
        
        system_prompt = f"""You are a database expert who helps select relevant tables for SQL queries.
        Given a user's question and available database schemas, select which tables are needed to answer the question.
        Only return the table names that are absolutely necessary to answer the question.

        {schema_info}

        Return your response as a JSON array of table names. For example: ["table1", "table2"]
        Do not include any explanation or other text."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0
            )
            
            selected_tables = json.loads(response.choices[0].message.content.strip())
            return selected_tables
            
        except Exception as e:
            st.error(f"Error in schema selection: {str(e)}")
            return []

# Query Generation Agent class
class QueryGenerationAgent:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
    
    def get_sample_data(self, schemas: List[Dict], conn: sqlite3.Connection) -> List[Dict]:
        """Get sample data for selected schemas"""
        tables_info = []
        for schema in schemas:
            table_name = schema['table_name']
            try:
                sample_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
                sample_data = []
                for _, row in sample_df.iterrows():
                    record = {}
                    for col, val in row.items():
                        if isinstance(val, str):
                            record[col] = val.encode().decode('unicode-escape')
                        else:
                            record[col] = val
                    sample_data.append(record)
                    
                tables_info.append({
                    "table_name": table_name,
                    "schema": schema['columns'],
                    "sample_data": sample_data
                })
            except Exception as e:
                print(f"Error getting sample data for {table_name}: {str(e)}")
                tables_info.append({
                    "table_name": table_name,
                    "schema": schema['columns'],
                    "sample_data": []
                })
        return tables_info

    def generate_query(self, user_question: str, selected_schemas: List[Dict], conn: sqlite3.Connection) -> str:
        """Generate SQL query for selected schemas"""
        tables_info = self.get_sample_data(selected_schemas, conn)
        
        system_prompt = "You are a SQL expert. Given the following schemas and sample data:\n\n"
        
        for table_info in tables_info:
            system_prompt += f"""
            Table: {table_info['table_name']}
            Columns:
            {json.dumps(table_info['schema'], indent=2, ensure_ascii=False)}

            Example data (first 3 rows):
            {json.dumps(table_info['sample_data'], indent=2, ensure_ascii=False)}

            """
        
        system_prompt += """
        Return Generate a SQL query to answer the user's question. The query should work with SQLite syntax.
        If the question requires joining tables, use appropriate JOIN statements.
        Return only the SQL query without any explanation."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"Error in query generation: {str(e)}")
            return ""


def create_table_from_df(df: pd.DataFrame, table_name: str, conn: sqlite3.Connection):
    """Create SQL table from DataFrame and insert data"""
    # Remove any invalid characters from table name and convert to lowercase
    safe_table_name = ''.join(c.lower() if c.isalnum() else '_' for c in table_name)
    
    try:
        # Write DataFrame to SQL
        df.to_sql(safe_table_name, conn, if_exists='replace', index=False)
        st.success(f"Table '{safe_table_name}' created successfully!")
        
        # Store the original and safe table name mapping in session state
        if 'table_mappings' not in st.session_state:
            st.session_state.table_mappings = {}
        st.session_state.table_mappings[table_name] = safe_table_name
        
        return safe_table_name
    except Exception as e:
        st.error(f"Error creating table: {str(e)}")

def execute_sql_query(query: str, conn: sqlite3.Connection):
    """Execute SQL query and return results"""
    try:
        # Show the actual query being executed for debugging
        st.subheader("Executing query:")
        st.code(query, language="sql")
        
        # For SELECT queries
        if query.strip().lower().startswith('select'):
            df = pd.read_sql_query(query, conn)
            if df.empty:
                st.warning("Query returned no results")
            return df
        # For other queries (INSERT, UPDATE, DELETE)
        else:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            st.success(f"Query affected {cursor.rowcount} rows")
            return pd.DataFrame({"message": [f"Query affected {cursor.rowcount} rows"]})
    except sqlite3.Error as e:
        st.error(f"SQLite Error: {str(e)}")
        # List available tables for debugging
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        st.text("Available tables in database:")
        st.code("\n".join([table[0] for table in tables]))
        return pd.DataFrame({"error": [str(e)]})
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return pd.DataFrame({"error": [str(e)]})

def get_table_schemas(db) -> List[Dict]:
    """Get all stored schemas from MongoDB"""
    collection = db["schemas"]
    return list(collection.find({}, {"_id": 0}))


def display_sidebar_menu(conn: sqlite3.Connection):
    """Display sidebar menu for database management"""
    with st.sidebar:
        st.header("Database Management")
        
        # Get list of tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            st.info("No tables in database")
            return
        
        st.subheader("Current Tables")
        
        # Display each table with remove button
        for table in tables:
            table_name = table[0]
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text(table_name)
            
            with col2:
                if st.button("🗑️", key=f"remove_{table_name}", help=f"Remove {table_name}"):
                    try:
                        # Confirm deletion
                        if st.session_state.get(f'confirm_delete_{table_name}', False):
                            # Delete from SQLite
                            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                            conn.commit()
                            
                            # Delete from MongoDB
                            db = connect_to_mongodb()
                            if db is not None:  # Fixed the comparison
                                collection = db["schemas"]
                                collection.delete_one({"table_name": table_name})
                            
                            st.success(f"Table '{table_name}' removed successfully!")
                            st.experimental_rerun()
                        else:
                            st.session_state[f'confirm_delete_{table_name}'] = True
                            st.warning(f"Click again to confirm removing '{table_name}'")
                            
                    except Exception as e:
                        st.error(f"Error removing table: {str(e)}")
                        
        # Add clear confirmation button states
        if st.button("Clear All Confirmation States"):
            for key in list(st.session_state.keys()):
                if key.startswith('confirm_delete_'):
                    del st.session_state[key]
            st.success("Confirmation states cleared")


def main():
    st.title("Multi-Agent SQL Query Assistant")

    schema_agent = SchemaSelectionAgent()
    query_agent = QueryGenerationAgent()
    
    # Initialize MongoDB connection
    db = connect_to_mongodb()
    
    if db is None:
        st.warning("Please start MongoDB service before using this application.")
        st.info("To start MongoDB, open Command Prompt as Administrator and run: net start MongoDB")
        return
    
    # Initialize SQLite connection
    conn = setup_sqlite_db()

    display_sidebar_menu(conn)
    
    # File upload section
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Read file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            table_name = st.text_input("Enter table name:", value=uploaded_file.name.split('.')[0].lower())
            
            if st.button("Process File"):
                # Generate and store schema
                schema = generate_sql_schema(df, table_name)
                store_schema_in_mongodb(schema, db)
                
                # Create SQL table and insert data
                if create_table_from_df(df, table_name, conn):
                    st.success(f"Schema for {table_name} has been stored and data loaded into SQL database!")
                
                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(df.head())
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Query section
    st.header("Ask Questions")
    
    # Get available schemas
    schemas = get_table_schemas(db)

    if schemas:
        # Table selection mode
        selection_mode = st.radio(
            "How would you like to select tables?",
            options=["Manual Selection", "AI Recommendation", "Use All Tables"],
            index=0,
            key="selection_mode"
        )
        
        if selection_mode == "Manual Selection":
            selected_table_names = st.multiselect(
                "Select tables to query:",
                options=[schema["table_name"] for schema in schemas]
            )
        
        user_question = st.text_input("Enter your question:")
        
        if user_question and st.button("Generate and Execute Query"):
            # Handle different selection modes
            if selection_mode == "Manual Selection":
                if not selected_table_names:
                    st.error("Please select at least one table.")
                    return
                selected_schemas = [
                    schema for schema in schemas 
                    if schema["table_name"] in selected_table_names
                ]
                st.write("Selected tables:", ", ".join(selected_table_names))
                
            elif selection_mode == "AI Recommendation":
                with st.spinner("AI is selecting relevant tables..."):
                    selected_table_names = schema_agent.select_schemas(user_question, schemas)
                    if not selected_table_names:
                        st.error("Could not determine which tables to use for this query.")
                        return
                    selected_schemas = [
                        schema for schema in schemas 
                        if schema["table_name"] in selected_table_names
                    ]
                    st.write("AI selected tables:", ", ".join(selected_table_names))
                    
            else:  # Use All Tables
                selected_schemas = schemas
                selected_table_names = [schema["table_name"] for schema in schemas]
                st.write("Using all tables:", ", ".join(selected_table_names))
            # First agent: Select relevant schemas
            with st.spinner("Selecting relevant tables..."):
                selected_table_names = schema_agent.select_schemas(user_question, schemas)
                
                if not selected_table_names:
                    st.error("Could not determine which tables to use for this query.")
                    return
                
                selected_schemas = [
                    schema for schema in schemas 
                    if schema["table_name"] in selected_table_names
                ]
                
                st.write("Selected tables:", ", ".join(selected_table_names))
            
            # Second agent: Generate SQL query
            with st.spinner("Generating SQL query..."):
                sql_query = query_agent.generate_query(user_question, selected_schemas, conn)
                
                if not sql_query:
                    st.error("Could not generate a valid SQL query.")
                    return
            
            # Execute query and display results
            results = execute_sql_query(sql_query, conn)
            
            st.subheader("Query Results:")
            if "error" in results.columns:
                st.error(results["error"][0])
            else:
                st.dataframe(results)
                
                # Download results option
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
    else:
        st.info("Please upload a file first to start querying!")
    
    # Close database connections
    conn.close()

if __name__ == "__main__":
    # Set up OpenAI API key
    openai.api_key = "OPEN_API_KEY"
    main()
