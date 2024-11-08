import sqlite3
import os


def create_database(db_name, path=os.getcwd()):
    conn = sqlite3.connect(os.path.join(path, f"{db_name}.db"))
    cursor = conn.cursor()

    # Create tables to store functions and arguments
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS functions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        start_line INTEGER,
        end_line INTEGER,
        file_path TEXT
    );
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS arguments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        function_id INTEGER NOT NULL,
        argument_type TEXT NOT NULL,
        name TEXT NOT NULL,
        FOREIGN KEY (function_id) REFERENCES functions (id) ON DELETE CASCADE
    );
    ''')

    conn.commit()
    conn.close()


def insert_function_data(functions_info, db_name, path=os.getcwd()):
    conn = sqlite3.connect(os.path.join(path, f"{db_name}.db"))
    cursor = conn.cursor()

    for function_info in functions_info:
        # Insert the function data
        cursor.execute(
            '''
            INSERT INTO functions (name, start_line, end_line, file_path)
            VALUES (?, ?, ?, ?)
            ''',
            (function_info['name'], function_info['start_line'],
             function_info['end_line'], function_info['file_path'])
        )

        # Get the ID of the last inserted function
        function_id = cursor.lastrowid

        # Insert function arguments (regular arguments)
        for arg in function_info['arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_id, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_id, 'argument', arg)
            )

        # Insert positional arguments
        for arg in function_info['positional_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_id, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_id, 'positional', arg)
            )

        # Insert variable-length positional arguments (if any)
        for arg in function_info['varlen_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_id, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_id, 'varlen', arg)
            )

        # Insert keyword arguments
        for arg in function_info['keyword_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_id, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_id, 'keyword', arg)
            )

        # Insert variable-length keyword arguments
        for arg in function_info['varlen_keyword_arguments']:
            cursor.execute(
                '''
                INSERT INTO arguments (function_id, argument_type, name)
                VALUES (?, ?, ?)
                ''',
                (function_id, 'varlen_keyword', arg)
            )

    conn.commit()
    conn.close()
