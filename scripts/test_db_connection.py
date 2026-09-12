import psycopg2
import config 

def test_connection():
    print(f"Connecting to database '{config.DB_NAME}' at {config.DB_HOST}:{config.DB_PORT}...")
    try:
        connection = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            sslmode=config.DB_SSLMODE,
            connect_timeout=10
        )
        cursor = connection.cursor()
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        
        if result == (1,):
            print("\nDatabase connection successful: SELECT 1 returned (1,)")
        
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"\nConnection failed: {e}")

if __name__ == "__main__":
    test_connection()