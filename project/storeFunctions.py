import sqlite3
import os

def create_database(db_name, path=os.getcwd()):
    conn = sqlite3.connect(path+'/'+db_name+'.db')
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS functions (
        name TEXT PRIMARY KEY NOT NULL,
        start_line INTEGER,
        end_line INTEGER
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS arguments (
        function_name TEXT NOT NULL,
        argument_type TEXT NOT NULL,
        name TEXT NOT NULL,
        default_value TEXT,
        FOREIGN KEY (function_name) REFERENCES functions (name) ON DELETE CASCADE
    );
    ''')

    conn.commit()

    conn.close()

def insert_function_data(functions_info, db_name, path=os.getcwd()):
    conn = sqlite3.connect(path+'/'+db_name+'.db')
    cursor = conn.cursor()

    for function_info in functions_info:
        # Insert the function data
        cursor.execute(
            '''
            INSERT INTO functions (name, start_line, end_line)
            VALUES (?, ?, ?)
            ''', 
            (function_info['name'], function_info['start_line'], function_info['end_line'])
        )

        # Get the function ID of the inserted function
        function_name = function_info['name']

        # Insert function arguments 
        for arg in function_info['arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_name, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_name, 'argument', arg)
            )        
        
        # Insert positional arguments
        for arg in function_info['positional_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_name, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_name, 'positional', arg)
            )

        # Insert variable-length positional arguments (if any)
        for arg in function_info['varlen_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_name, argument_type, name)
                VALUES (?, ?, ?)
                ''', 
                (function_name, 'varlen', arg)
            )

        # Insert keyword arguments
        for arg in function_info['keyword_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_name, argument_type, name)
                VALUES (?, ?, ?)
                ''', 
                (function_name, 'keyword', arg)
            )

        # Insert variable-length keyword arguments
        for arg in function_info['varlen_keyword_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_name, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_name, 'varlen_keyword', arg)
            )
        
    conn.commit()

    conn.close()